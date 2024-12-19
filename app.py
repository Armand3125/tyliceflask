from flask import Flask, render_template, request, send_file
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import base64
from datetime import datetime
import os

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

# Fonction pour convertir l'image en Base64
def encode_image_base64(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_colors = []
    selected_color_names = []
    num_selections = 4

    if request.method == 'POST':
        uploaded_file = request.files.get('image')
        num_selections = int(request.form.get('num_selections', 4))

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            width, height = image.size
            dim = 350
            new_width = dim if width > height else int((dim / height) * width)
            new_height = dim if height >= width else int((dim / width) * height)
            resized_image = image.resize((new_width, new_height))
            img_arr = np.array(resized_image)

            if img_arr.shape[-1] == 3:
                pixels = img_arr.reshape(-1, 3)
                kmeans = KMeans(n_clusters=num_selections, random_state=0).fit(pixels)
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

                centers_rgb = np.array(centers, dtype=int)
                pal_rgb = np.array(list(pal.values()), dtype=int)
                distances = np.linalg.norm(centers_rgb[:, None] - pal_rgb[None, :], axis=2)

                ordered_colors_by_cluster = []
                for i in range(num_selections):
                    closest_colors_idx = distances[i].argsort()
                    ordered_colors_by_cluster.append([list(pal.keys())[idx] for idx in closest_colors_idx])

                cluster_counts = np.bincount(labels)
                total_pixels = len(labels)
                cluster_percentages = (cluster_counts / total_pixels) * 100

                sorted_indices = np.argsort(-cluster_percentages)
                sorted_percentages = cluster_percentages[sorted_indices]
                sorted_ordered_colors_by_cluster = [ordered_colors_by_cluster[i] for i in sorted_indices]

                # Get selected colors
                for i, cluster_index in enumerate(sorted_indices):
                    color_name = request.form.get(f"color_select_{i}")
                    selected_colors.append(pal.get(color_name))
                    selected_color_names.append(color_name)

                # Recréer l'image avec les nouvelles couleurs
                new_img_arr = np.zeros_like(img_arr)
                for i in range(img_arr.shape[0]):
                    for j in range(img_arr.shape[1]):
                        lbl = labels[i * img_arr.shape[1] + j]
                        new_color_index = np.where(sorted_indices == lbl)[0][0]
                        new_img_arr[i, j] = selected_colors[new_color_index]

                new_image = Image.fromarray(new_img_arr.astype('uint8'))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"{''.join(selected_color_names)}_{timestamp}.png"
                img_base64 = encode_image_base64(new_image)
                
                return render_template('index.html', image=new_image, img_base64=img_base64, file_name=file_name, 
                                       selected_color_names=selected_color_names, num_selections=num_selections)
    
    return render_template('index.html', selected_colors=selected_colors, num_selections=num_selections)

@app.route('/download/<file_name>', methods=['GET'])
def download(file_name):
    file_path = os.path.join('static', 'images', file_name)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
