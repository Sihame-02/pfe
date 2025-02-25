import cv2


def start_camera():
    # Ouvrir la caméra
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Impossible d'accéder à la caméra.")
        return

    print("[INFO] Appuyez sur 'q' pour quitter.")

    while True:
        # Lire une image depuis la caméra
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Impossible d'afficher la caméra.")
            break

        # Afficher l'image dans une fenêtre
        cv2.imshow('Flux en temps réel', frame)

        # Quitter en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la caméra
    cap.release()
    cv2.destroyAllWindows()
