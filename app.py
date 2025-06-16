from flask import Flask, render_template, request
import requests
import os
from datetime import datetime
from azure.storage.blob import BlobClient
from urllib.parse import quote_plus

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== Custom Vision Credentials ====
PREDICTION_KEY = "EDCh7x6CiNLPyKe7xhDBAO1ipWp1y2W1zHFkPQG4bxlS135aZ7pNJQQJ99BFACLArgHXJ3w3AAAIACOG3z5i"
ENDPOINT = "https://vrutcustomv-prediction.cognitiveservices.azure.com" #ENDPOINT
PROJECT_ID = "b9be6885-76e7-478d-af52-4cb08192b508"   #project id
ITERATION_NAME = "Iteration1"   #itration name
PREDICTION_URL = f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{ITERATION_NAME}/image"

# ==== Azure Blob SAS Settings ====
AZURE_STORAGE_ACCOUNT = "vrutprojectsa"    #name of project account
AZURE_CONTAINER_NAME = "images"
SAS_TOKEN = "?sp=rcw&st=2025-06-11T16:03:18Z&se=2025-06-12T00:03:18Z&spr=https&sv=2024-11-04&sr=c&sig=Q5hcrclyAon5ARdBVoNTUlRO1vMB3d6pwiz%2FgEJupl4%3D"

def upload_to_blob_with_sas(image_path, blob_name):
    try:
        blob_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{quote_plus(blob_name)}{SAS_TOKEN}"
        blob_client = BlobClient.from_blob_url(blob_url)

        with open(image_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f" Uploaded to: {blob_url}")
        return blob_url

    except Exception as e:
        print(" Blob upload error:", e)
        return None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Upload to Azure Blob
    blob_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    blob_url = upload_to_blob_with_sas(image_path, blob_name)

    # Predict with Custom Vision
    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream"
    }

    try:
        with open(image_path, "rb") as image_data:
            image_bytes = image_data.read()

        response = requests.post(PREDICTION_URL, headers=headers, data=image_bytes)
        response.raise_for_status()

        predictions = response.json().get("predictions", [])
        if not predictions:
            return render_template("index.html", error="No prediction found.", blob_url=blob_url)

        top_prediction = predictions[0]
        tag = top_prediction["tagName"]
        confidence = top_prediction["probability"]

        if confidence >= 0.7:
            return render_template("index.html", prediction=tag, blob_url=blob_url)
        else:
            return render_template("index.html", error="Image is not in the scope", blob_url=blob_url)

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

    finally:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete file: {cleanup_error}")

if __name__ == "__main__":
    app.run(debug=True)
