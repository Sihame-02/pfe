import cv2
import pytesseract


def extract_text(image):
    """Extrait le texte d'une image avec Tesseract OCR après prétraitement."""

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuillage pour améliorer la détection du texte
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Utilisation de Tesseract OCR pour extraire le texte
    texte_extrait = pytesseract.image_to_string(thresh, lang="fra")

    return texte_extrait
