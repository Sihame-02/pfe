import together
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Définir la clé API depuis l'environnement
together.api_key = os.getenv("TOGETHER_API_KEY")

# Vérifier si la clé API est bien définie
if not together.api_key:
    print("❌ Clé API TogetherAI manquante ! Vérifie ton fichier .env")

user_message = "hello, user"  # Message utilisateur

# Utilisation de la nouvelle syntaxe de Together
response = together.Complete.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            prompt=user_message,
            max_tokens=150,
            temperature=0.7  # Un peu de variabilité dans la réponse
        )
# Afficher la réponse brute pour debug
print(response)

# Extraire correctement le texte généré
if "choices" in response and len(response["choices"]) > 0:
    print(response["choices"][0]["message"]["content"].strip())
else:
    print("Erreur : Aucune réponse générée.")