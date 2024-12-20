import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Palette de couleurs
pal = {
    "Noir": (0, 0, 0), "Blanc": (255, 255, 255),
    "Or": (228, 189, 104), "Cyan": (0, 134, 214),
    "Lila": (174, 150, 212), "Vert": (63, 142, 67),
    "Rouge": (222, 67, 67), "Bleu": (0, 120, 191),
    "Orange": (249, 153, 99), "Vert foncé": (59, 102, 94),
    "Bleu clair": (163, 216, 225), "Magenta": (236, 0, 140),
    "Argent": (166, 169, 170), "Violet": (94, 67, 183),
    "Bleu foncé": (4, 47, 86),
}

@app.route("/")
def home():
    return "Hello, Flask on Heroku! Upload an image to get started."

@app.route("/process", methods=["POST"])
def process_image():
    try:
        # Vérification du fichier envoyé
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        image = Image.open(file).convert("RGB")

        # Paramètres par défaut
        num_selections = int(request.form.get("num_colors", 4))
        image = image.resize((350, 350))  # Redimensionne pour le traitement

        # KMeans pour la sélection des couleurs
        img_arr = np.array(image)
        pixels = img_arr.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
        centers = np.array(kmeans.cluster_centers_, dtype=int)

        # Conversion des couleurs
        pal_rgb = np.array(list(pal.values()), dtype=int)
        distances = np.linalg.norm(centers[:, None] - pal_rgb[None, :], axis=2)

        selected_colors = []
        for i in range(num_selections):
            closest_color_idx = distances[i].argmin()
            selected_colors.append(list(pal.keys())[closest_color_idx])

        # Retourne les couleurs dominantes
        return jsonify({
            "colors": selected_colors
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Utilisation du port assigné par Heroku
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
