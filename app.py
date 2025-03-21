from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import os
import base64
import numpy as np
import pyttsx3
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from text_recognition import extract_and_speak_text

app = Flask(__name__)

# Configuration du dossier pour télécharger les fichiers
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Historique des fichiers téléchargés
historique_fichiers = []

# Configuration de la caméra
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERREUR] Impossible d'accéder à la caméra.")
    cap = None

# Initialisation du moteur TTS
engine = pyttsx3.init()

# Charger Faster R-CNN pour la détection d'objets
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
transform = T.Compose([T.ToTensor()])

# Fonction de génération des images pour le flux vidéo
def generate_frames():
    if cap is None:
        return  # Si la caméra ne fonctionne pas, ne pas continuer
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        extract_and_speak_text(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    if cap is None:
        return "[ERREUR] Caméra non accessible"
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/aide')
def aide():
    return render_template('aide.html')

@app.route('/historique')
def historique():
    fichiers_disponibles = os.listdir(app.config["UPLOAD_FOLDER"])
    return render_template('historique.html', fichiers=fichiers_disponibles)

@app.route('/upload', methods=['POST'])
def upload_file():
    if "file" not in request.files:
        print("[ERREUR] Aucun fichier sélectionné.")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        print("[ERREUR] Nom de fichier vide.")
        return redirect(request.url)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    historique_fichiers.append(file.filename)
    print(f"[INFO] Fichier enregistré : {file.filename} dans {file_path}")
    return redirect(url_for('historique'))

@app.route('/effacer_historique', methods=['POST'])
def effacer_historique():
    global historique_fichiers
    historique_fichiers.clear()
    for fichier in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, fichier))
    return redirect(url_for('historique'))

@app.route("/detect_text", methods=["POST"])
def detect_text():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    text = extract_and_speak_text(image)
    return jsonify({"text": text})

@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    detected_objects = detect_objects_in_frame(image)
    objects = [obj[1] for obj in detected_objects]
    return jsonify({"objects": objects})

# Fonction pour la détection d'objets
def detect_objects_in_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)
    detected_objects = []
    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > 0.5:
            label_id = predictions[0]['labels'][i].item()
            object_definition = get_object_definition(label_id)
            detected_objects.append((label_id, object_definition))
    return detected_objects

# Fonction pour obtenir les définitions des objets détectés
def get_object_definition(label_id):
    object_definitions = {
        1: "Person: Un être humain.",
        2: "Bicycle: Un véhicule à deux roues propulsé par la force humaine.",
        3: "Car: Un véhicule à moteur, généralement à quatre roues.",
        4: "Dog: Un mammifère domestiqué, souvent un animal de compagnie.",
        5: "Cat: Un petit mammifère domestiqué, souvent un animal de compagnie."
    }
    return object_definitions.get(label_id, "Objet inconnu.")

if __name__ == '__main__':
    app.run(debug=True)
