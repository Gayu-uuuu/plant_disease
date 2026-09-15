from flask import Flask, render_template, request
from transformers import pipeline
from PIL import Image
import os

app = Flask(__name__)

# Create upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained plant disease model
print("Loading plant disease model...")
classifier = pipeline(
    "image-classification",
    model="Kathir56/plant-disease-tamilnadu"
)
print("Model loaded successfully!")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            # Save uploaded image
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # Open image
            image = Image.open(image_path).convert("RGB")

            # Predict top 3 diseases
            prediction = classifier(image, top_k=3)

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)