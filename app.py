from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_wtf.file import FileRequired, FileAllowed
from flask_uploads import UploadSet, IMAGES, configure_uploads
from werkzeug.utils import secure_filename
import os
import main  # Ensure main.py contains the process_image function
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Create 'uploads' folder if it doesn't exist
path = 'static/temp'


app.config['UPLOADED_PHOTOS_DEST'] = path
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload Sudoku Image')


@app.route(f'/{path}/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route("/", methods=["GET", "POST"])
def upload_image():
    # Remove all cache
    if os.listdir(path):
        for name in os.listdir(path):
            file = f"{path}/{name}"
            if os.path.isfile(file):
                os.remove(file)

    # Create an instance of the form class
    form = UploadForm()
    # Validate the upload of image
    if form.validate_on_submit():
        # Save the file (image)
        filename = photos.save(form.photo.data)
        # Get the original image
        file_url = url_for('get_file', filename=filename)
        # Solved sudoku
        images, solved_sudoku = main.process_image(filename)
        # Save every url of the image in a variable
        images_url = [url_for('get_file', filename=image) for image in images]
        # Convert solved_sudoku to list
        solved_sudoku_list = solved_sudoku.tolist()
    else:
        file_url, images_url, solved_sudoku_list = None, None, None

    return render_template("index.html",
                           form=form,
                           file_url=file_url,
                           images=images_url,
                           solution=solved_sudoku_list)
  # Pass NumPy array directly


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/capture', methods=['POST'])
def capture():
    # Pastikan variabel path terdefinisi
    if not os.path.exists(path):
        os.makedirs(path)

    # Remove all cache
    if os.listdir(path):
        for name in os.listdir(path):
            file = f"{path}/{name}"
            if os.path.isfile(file):
                os.remove(file)

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data found in request'}), 400

    # Mengambil data gambar dari kamera
    try:
        image_data = data['image'].split(',')[1]
        image_data = base64.b64decode(image_data)
    except (IndexError, ValueError) as e:
        return jsonify({'error': 'Invalid image data format'}), 400

    # Save the image with a fixed name
    filename = 'captured_image.jpg'
    filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
    try:
        with open(filepath, 'wb') as f:
            f.write(image_data)  # Simpan gambar ke server
    except IOError as e:
        return jsonify({'error': f'Failed to save image: {str(e)}'}), 500

    app.logger.debug(f'Captured image saved to {filename}')

    # Process the captured image to solve the Sudoku
    try:
        # Solved sudoku
        images, solved_sudoku = main.process_image_cap(filename)
        # Save URLs of processed images
        images_url = [url_for('get_file', filename=image) for image in images]
        solved_sudoku_list = solved_sudoku.tolist()

        # Return data in the same way as upload_image
        return jsonify({
            'file_url': url_for('get_file', filename=filename),
            'images': images_url,
            'solution': solved_sudoku_list
        })

    except Exception as e:
        app.logger.error(f'Error processing captured image: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
