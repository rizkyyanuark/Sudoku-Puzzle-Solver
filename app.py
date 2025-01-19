from flask import Flask, request, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from util.main import process_image, process_image_cap
import base64

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS untuk semua rute

app.config['SECRET_KEY'] = os.urandom(24)
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
path = 'static/temp'
if not os.path.exists(path):
    os.makedirs(path)
app.config['UPLOAD_FOLDER'] = path


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Welcome to model api sudoku solver",
        },
        "data": None
    }), 200


@app.route('/upload', methods=['POST'])
def upload_image():
    if os.listdir(path):
        for name in os.listdir(path):
            file = f"{path}/{name}"
            if os.path.isfile(file):
                os.remove(file)
    if "image" not in request.files:
        return jsonify({
            "status": {
                "code": 400,
                "message": "No image provided",
            },
            "data": None
        }), 400

    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        images, solved_sudoku = process_image(filename)
        image_paths = [url_for('uploaded_file', filename=image)
                       for image in images]
        solved_sudoku_list = solved_sudoku.tolist() if solved_sudoku is not None else []
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success",
            },
            "data": {
                "images": image_paths,
                "solution": solved_sudoku_list
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Invalid file type",
            },
            "data": None
        }), 400


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

    filename = 'captured_image.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_data)

    images, solved_sudoku = process_image_cap(filename)
    image_paths = [url_for('uploaded_file', filename=image)
                   for image in images]
    solved_sudoku_list = solved_sudoku.tolist() if solved_sudoku is not None else []
    return jsonify({
        'images': image_paths,
        'solution': solved_sudoku_list
    }), 200


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
