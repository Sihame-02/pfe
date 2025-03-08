import cv2
import pytesseract
import pyttsx3

# Configurez le chemin de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(frame):
    """
    Pré-traite l'image pour améliorer la détection de texte.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(
        gray, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    return gray


def extract_and_speak_text(frame):
    """
    Fonction pour extraire le texte de l'image et le lire à voix haute.
    """
    processed_frame = preprocess_image(frame)

    # Debug : Afficher l'image traitée
    cv2.imshow("Image traitée", processed_frame)

    # Extraire le texte avec Tesseract
    detected_text = pytesseract.image_to_string(processed_frame, lang='eng').strip()

    if detected_text:
        print(f"[TEXTE DÉTECTÉ] : {detected_text}")
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Ajuster la vitesse de la voix
        engine.say(detected_text)
        engine.runAndWait()
    else:
        print("Aucun texte détecté.")
