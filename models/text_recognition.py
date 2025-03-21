import cv2
import pytesseract
import pyttsx3

engine = pyttsx3.init()

def extract_and_speak_text(image):
    """Detecte et lit le texte à haute voix à partir de l'image donnée."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    text = pytesseract.image_to_string(gray, lang="eng+fra")

    if text.strip():
        print("Texte détecté :", text)
        engine.say(text)
        engine.runAndWait()
    else:
        print("Aucun texte détecté.")

    return text
