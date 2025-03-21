from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import together
import os
import traceback
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configurer l'upload des fichiers
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Définir la clé API TogetherAI
together.api_key = os.getenv("TOGETHER_API_KEY")

if not together.api_key:
    print("❌ Clé API TogetherAI manquante ! Vérifie ton fichier .env")

@app.route('/')
def test():
    return render_template('test.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Le message est vide"}), 400

    try:
        # Appel à l'API TogetherAI
        response = together.Complete.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            prompt=user_message,
            max_tokens=150,
            temperature=0.7
        )

        # Vérifier et extraire correctement la réponse
        if "choices" in response and len(response["choices"]) > 0:
            bot_response = response["choices"][0]["text"].strip()
        else:
            bot_response = "Désolé, je n'ai pas compris."

        return jsonify({"response": bot_response})

    except Exception as e:
        print("❌ Erreur API TogetherAI :", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    return jsonify({"message": "Fichier reçu !", "url": f"/uploads/{file.filename}"})

@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)