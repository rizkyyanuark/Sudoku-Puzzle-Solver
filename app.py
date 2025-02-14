from flask import Flask, request, url_for, jsonify, session
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from util.main import process_image, process_image_cap
import base64
import json
import firebase_admin
from firebase_admin import credentials, storage
from google.cloud import secretmanager

app = Flask(__name__)
CORS(app)


def get_secret(secret_name, project_id=None):
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("PROJECT_ID")
    secret_version = f'projects/{project_id}/secrets/{secret_name}/versions/latest'
    response = client.access_secret_version(name=secret_version)
    return response.payload.data.decode('UTF-8')


firebase_key = get_secret("sudoku-solver")
firebase_key = json.loads(firebase_key)

# Initialize Firebase Admin SDK
cred = credentials.Certificate(firebase_key)
firebase_admin.initialize_app(cred, {
    "storageBucket": firebase_key["bucket-firestore"]
})


app.config['SECRET_KEY'] = os.urandom(24)
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
app.config['UPLOAD_FOLDER'] = 'static/temp'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def upload_to_firebase(file, filename):
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_file(file)
    blob.make_public()
    return blob.public_url


def delete_from_firebase(filename):
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.delete()


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Welcome to model api sudoku solver",
        },
        "data": None
    })


@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file and allowed_file(file.filename):
        if 'previous_image' in session:
            delete_from_firebase(session['previous_image'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        original_image_url = upload_to_firebase(
            open(file_path, 'rb'), filename)

        images, solved_sudoku = process_image(filename)
        image_urls = [upload_to_firebase(open(os.path.join(app.config['UPLOAD_FOLDER'], image), 'rb'), image)
                      for image in images]

        session['previous_image'] = filename

        solved_sudoku_list = solved_sudoku.tolist() if solved_sudoku is not None else []
        return jsonify({
            "status": {
                "code": 200,
                "message": "Image processed successfully"
            },
            "data": {
                "original_image": original_image_url,
                "images": image_urls,
                "solution": solved_sudoku_list
            }
        })
    else:
        return jsonify({"error": "Invalid file type"}), 400


@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data found in request'}), 400

    try:
        image_data = data['image'].split(',')[1]
        image_data = base64.b64decode(image_data)
    except (IndexError, ValueError):
        return jsonify({'error': 'Invalid image data format'}), 400

    if 'previous_image' in session:
        delete_from_firebase(session['previous_image'])

    filename = 'captured_image.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_data)

    original_image_url = upload_to_firebase(open(filepath, 'rb'), filename)

    images, solved_sudoku = process_image_cap(filename)
    image_urls = [upload_to_firebase(open(os.path.join(app.config['UPLOAD_FOLDER'], image), 'rb'), image)
                  for image in images]

    session['previous_image'] = filename

    solved_sudoku_list = solved_sudoku.tolist() if solved_sudoku is not None else []
    return jsonify({
        "status": {
            "code": 200,
            "message": "Image processed successfully"
        },
        "data": {
            "original_image": original_image_url,
            "images": image_urls,
            "solution": solved_sudoku_list
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
