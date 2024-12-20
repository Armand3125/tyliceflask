import os
from flask import Flask, request, render_template, jsonify, send_file
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
from datetime import datetime
import base64

app = Flask(__name__)

# Palette de couleurs
pal = {
    "NC": (0, 0, 0), "BJ": (255, 255, 255),
    "JO": (228, 189, 104), "BC": (0, 134, 214),
    "VL": (174, 150, 212), "VG": (63, 142, 67),
    "RE": (222, 67, 67), "BM": (0, 120, 191),
    "OM": (249, 153, 99), "VGa": (59, 102, 94),
    "BG": (163, 216, 225), "VM": (236, 0, 140),
    "GA": (166, 169, 170), "VB": (94, 67, 183),
    "BF": (4, 47, 86),
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    try:
        # Récupérer l'image et les données depuis le formulaire
        file = request.files.get("file")
        num_selections = int(request.form.get("num_colors", 4))
        if not file:
            return jsonify({"error": "Aucune image n'a été envoyée"}), 400

        image = Image.open(file).convert("RGB")
        width, height = image.size
        dim = 350
        new_width = dim if width > height else int((dim / height) * width)
        new_height = dim if height >= width else int((dim / width) * height)

        resized_image = image.resize((new_width, new_height))
        img_arr = np.array(resized_image)
        pixels = img_arr.reshape(-1, 3)

        # KMeans clustering
        kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
        labels = kmeans.labels_
        centers = np.array(kmeans.cluster_centers_, dtype=int)

        # Associer les couleurs aux noms
        pal_rgb = np.array(list(pal.values()), dtype=int)
        distances = np.linalg.norm(centers[:, None] - pal_rgb[None, :], axis=2)

        selected_colors = []
        for i in range(num_selections):
            closest_color_idx = distances[i].argmin()
            selected_colors.append(list(pal.keys())[closest_color_idx])

        # Recréer l'image avec les nouvelles couleurs
        new_img_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                lbl = labels[i * img_arr.shape[1] + j]
                new_img_arr[i, j] = centers[lbl]

        new_image = Image.fromarray(new_img_arr.astype('uint8'))

        # Convertir l'image en base64 pour affichage
        img_io = io.BytesIO()
        new_image.save(img_io, format="PNG")
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.read()).decode()

        # Retourner les données
        return render_template(
            "results.html",
            original_img=file.filename,
            new_img_base64=img_base64,
            selected_colors=selected_colors,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
