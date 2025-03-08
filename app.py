from flask import Flask, render_template, Response
import cv2
import pyttsx3
import pytesseract
import os
from text_recognition import extract_and_speak_text
# Decommenter ou ajouter cette ligne seulement quand la fonction detect_objects est prête
# from object_detection import detect_objects

app = Flask(__name__)

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Fonction pour générer le flux vidéo
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détecter et extraire du texte en temps réel
        extract_and_speak_text(frame)

        # Convertir l'image en format JPEG pour l'affichage dans la page web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """
    Route d'accueil avec le menu de l'application.
    """
    return render_template('index.html')

@app.route('/camera')
def camera():
    """
    Route pour afficher la page de la caméra avec les boutons pour détecter du texte ou des objets.
    """
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    """
    Route pour diffuser le flux vidéo en temps réel.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings')
def settings():
    """
    Route pour afficher les paramètres de l'application.
    """
    return render_template('settings.html')

@app.route('/about')
def about():
    """
    Route pour afficher les informations sur l'application.
    """
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
