import torch
import torchvision.transforms as T
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Charger Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transformation d'image
transform = T.Compose([T.ToTensor()])

# Dictionnaire pour définir des objets (ajouter plus de descriptions si nécessaire)
object_definitions = {
    1: "Person: Un être humain.",
    2: "Bicycle: Un véhicule à deux roues propulsé par la force humaine.",
    3: "Car: Un véhicule à moteur, généralement à quatre roues.",
    4: "Dog: Un mammifère domestiqué, souvent un animal de compagnie.",
    5: "Cat: Un petit mammifère domestiqué, souvent un animal de compagnie."
    # Ajouter d'autres objets ici...
}

# Fonction pour récupérer la définition de l'objet
def get_object_definition(label_id):
    return object_definitions.get(label_id, "Objet inconnu.")

# Accès à la webcam
cap = cv2.VideoCapture(0)

def detect_objects_in_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).unsqueeze(0)

    # Détection
    with torch.no_grad():
        predictions = model(img_tensor)

    detected_objects = []
    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > 0.5:
            label_id = predictions[0]['labels'][i].item()
            object_definition = get_object_definition(label_id)
            detected_objects.append((label_id, object_definition))
    return detected_objects, frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detected_objects, frame = detect_objects_in_frame(frame)

    # Dessiner les boîtes
    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Obj {predictions[0]['labels'][i].item()} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
