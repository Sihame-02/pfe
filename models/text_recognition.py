import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os

# Configurez le chemin vers Tesseract si nécessaire
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image):
    """
    Prétraiter l'image pour améliorer la reconnaissance OCR.
    """
    gray_image = image.convert("L")  # Conversion en niveaux de gris
    resized_image = gray_image.resize((gray_image.size[0] * 2, gray_image.size[1] * 2))  # Zoom x2
    enhanced_image = ImageEnhance.Contrast(resized_image).enhance(2)  # Augmenter le contraste
    return enhanced_image


def main():
    # Chemin de l'image
    image_path = r"C:\Users\USER\PycharmProjects\PythoProject\data\synthtext\samples.png"

    # Vérifiez si le fichier existe
    if not os.path.exists(image_path):
        print(f"[ERROR] L'image '{image_path}' est introuvable.")
        return

    try:
        # Charger l'image
        image = Image.open(image_path)

        # Prétraiter l'image
        print("[INFO] Prétraitement de l'image...")
        processed_image = preprocess_image(image)
        processed_image.show()  # Ouvrir une fenêtre montrant l'image prétraitée

        # Extraire le texte
        print("[INFO] Extraction du texte...")
        extracted_text = pytesseract.image_to_string(processed_image, lang="fra+ara+eng")
        if extracted_text.strip():
            print(f"Texte extrait :\n{extracted_text}")
        else:
            print("[INFO] Aucun texte détecté dans l'image. Vérifiez la qualité de l'image.")
    except Exception as e:
        print(f"[ERROR] Une erreur s'est produite : {e}")


if __name__ == "__main__":
    main()
