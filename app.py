from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import base64
import io
from datetime import datetime

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

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/process', methods=["POST"])
def process():
    uploaded_file = request.files['image']
    num_colors = int(request.form.get("num_colors", 4))

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_arr = np.array(image)
        pixels = img_arr.reshape(-1, 3)

        # KMeans clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Associer les couleurs aux clusters
        centers_rgb = np.array(centers, dtype=int)
        pal_rgb = np.array(list(pal.values()), dtype=int)
        distances = np.linalg.norm(centers_rgb[:, None] - pal_rgb[None, :], axis=2)

        ordered_colors_by_cluster = []
        for i in range(num_colors):
            closest_colors_idx = distances[i].argsort()
            ordered_colors_by_cluster.append([list(pal.keys())[idx] for idx in closest_colors_idx])

        # Calcul des proportions
        cluster_counts = np.bincount(labels)
        total_pixels = len(labels)
        cluster_percentages = (cluster_counts / total_pixels) * 100
        sorted_indices = np.argsort(-cluster_percentages)

        selected_colors = []
        for i, cluster_index in enumerate(sorted_indices):
            selected_colors.append(ordered_colors_by_cluster[i][0])

        # Générer une nouvelle image
        new_img_arr = np.zeros_like(img_arr)
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                lbl = labels[i * img_arr.shape[1] + j]
                new_color_index = np.where(sorted_indices == lbl)[0][0]
                new_img_arr[i, j] = pal[selected_colors[new_color_index]]

        new_image = Image.fromarray(new_img_arr.astype('uint8'))

        # Convertir en base64 pour affichage
        img_io = io.BytesIO()
        new_image.save(img_io, format="PNG")
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return render_template(
            "results.html",
            original_img=uploaded_file.filename,
            new_img_base64=img_base64,
            selected_colors=selected_colors,
            pal=pal,
        )
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
