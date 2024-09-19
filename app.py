from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_wtf.file import FileRequired, FileAllowed
from werkzeug.utils import secure_filename
import os
import logging
import main  # Ensure main.py contains the process_image function

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create 'uploads' folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set up logging
logging.basicConfig(level=logging.DEBUG)


class UploadForm(FlaskForm):
    photo = FileField('Upload Image', validators=[
                      FileRequired(), FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])
    submit = SubmitField('Upload and Solve')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        # Save the uploaded file
        file = form.photo.data
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        new_filename = f'original_image{file_ext}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        # Save the file with the new name
        file.save(filepath)

        app.logger.debug(f'File {new_filename} saved to {filepath}')

        # Process the image and solve the Sudoku
        try:
            result = main.process_image(filepath)
            return render_template('index.html', form=form, file_url=url_for('uploaded_file', filename=new_filename),
                                   images=result['images'], solution=result['solution'])
        except Exception as e:
            app.logger.error(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))

    return render_template('index.html', form=form)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
