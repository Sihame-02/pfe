import cv2
import pytesseract
import pyttsx3

# Configurez le chemin de Tesseract (modifiez selon votre installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(frame):
    """
    Pré-traite l'image pour améliorer la détection de texte.
    """
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuil adaptatif pour améliorer la visibilité des caractères
    gray = cv2.adaptiveThreshold(
        gray, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    return gray


def start_text_recognition():
    """
    Démarre la reconnaissance de texte en temps réel avec la caméra
    et lit les textes détectés à haute voix.
    """
    # Initialiser le moteur de synthèse vocale
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjuster la vitesse de la voix

    print("[INFO] Début de la reconnaissance de texte (Appuyez sur 'q' pour quitter).")

    # Ouvrir la caméra
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Impossible d'accéder à la caméra.")
        return

    previous_text = ""  # Stocke le dernier texte détecté pour éviter les répétitions

    while True:
        # Lire une image depuis la caméra
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Impossible de capturer l'image.")
            break

        # Pré-traiter l'image pour améliorer la reconnaissance de texte
        processed_frame = preprocess_image(frame)

        # Extraire le texte avec Tesseract
        detected_text = pytesseract.image_to_string(processed_frame, lang='eng').strip()

        # Vérifier si du texte a été détecté et éviter les doublons
        if detected_text and detected_text != previous_text:
            print("[TEXTE DÉTECTÉ] :", detected_text)  # Afficher le texte détecté dans la console
            engine.say(detected_text)  # Lire le texte à voix haute
            engine.runAndWait()  # Exécuter la synthèse vocale
            previous_text = detected_text  # Mettre à jour le texte précédent

        # Afficher la vidéo en temps réel
        cv2.imshow("Reconnaissance de Texte", frame)

        # Arrêter quand on appuie sur la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération des ressources
    camera.release()
    cv2.destroyAllWindows()
    print("[INFO] Reconnaissance de texte terminée.")
