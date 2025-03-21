import cv2
import pytesseract
import pyttsx3

# Initialisation du moteur de synthèse vocale
engine = pyttsx3.init()


def extract_and_speak_text(image):
    """
    Détecte le texte dans une image et le lit à haute voix.

    :param image: Image capturée (numpy array)
    :return: Texte détecté
    """
    # Conversion en niveaux de gris pour améliorer la reconnaissance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Application d'un filtre pour améliorer la détection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Extraction du texte avec Tesseract OCR
    text = pytesseract.image_to_string(gray, lang="eng+fra")  # Anglais + Français

    if text.strip():  # Vérifier que du texte a été détecté
        print("Texte détecté :", text)
        engine.say(text)  # Lecture à haute voix
        engine.runAndWait()
    else:
        print("Aucun texte détecté.")

    return text


def detect_text_from_image_file(image_path):
    """
    Charge une image depuis un fichier et applique la détection de texte.

    :param image_path: Chemin du fichier image
    :return: Texte détecté
    """
    image = cv2.imread(image_path)  # Charger l'image depuis le fichier
    if image is None:
        print("Erreur : Impossible de charger l'image.")
        return None

    return extract_and_speak_text(image)


if __name__== "_main_":
    # Test avec une image locale (remplace "image.jpg" par le chemin de ton image)
    detect_text_from_image_file("image.jpg")