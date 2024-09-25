<div align="center">
  <h1>Sudoku Solver Dengan Digital Image Processing dan CNN</h1>
  <img src="https://github.com/rizkyyanuark/SudokuSolver-DataCitra/blob/main/util/sudoku.png" align="center" alt="Logo" width="275" height="275">
  <blockquote>
This project employs two approaches: digital image processing and convolutional neural networks (CNNs). Digital image processing is used for image extraction, while CNNs are used for learning and prediction.
    <br>
The main goal of this project is to come up with an algorithm that can solve Sudoku puzzles from image-based input in an efficient and accurate way.
  </blockquote>
</div>


## ✨ Features

- **📤 Upload Image** : Users can upload an image of the Sudoku board from their device, which will be processed and solved by the application.
- **📸 Use Camera** : Users can utilize their device's camera to capture an image of the Sudoku board in real-time, offering a convenient way to scan the board instantly.
- **🤖 Automatic Detection & Solving** : The app automatically detects the Sudoku grid, recognizes the numbers using a machine learning model, and solves the puzzle for you.
- **📊 Result Display** : After solving, the original image, the processed image, and the solved Sudoku puzzle are presented in an easy-to-read table format.

## 🛠️ Technologies Used
- **Frontend** : HTML, CSS, JavaScript, Bootstrap
- **Backend** : Python, Flask
- **Image Processing** : OpenCV
- **Machine Learning** : TensorFlow, Keras
- **OCR Model** : EasyOCR

## 🚀 Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/Sudoku-Puzzle-Solver.git
   cd Sudoku-Puzzle-Solver
2. **Create and activate a virtual environment:**
   - On Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - on Macos/Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```
   python app.py
   ```
5. **Access the application in your browser:**


## Project Structure

```
sudoku-solver/
├── app.py                     # Main Flask application to run the web server
├── main.py                    # Entry point of the application, handling routing and logic
├── sudoku_solver.py           # Core logic for solving the Sudoku puzzle using algorithms or ML model
├── utils.py                   # Utility functions used throughout the project (image processing, etc.)
├── templates/                 # HTML templates for rendering the web pages
│   └── index.html             # Main page for the Sudoku solver web interface
├── static/                    # Static files such as CSS, images, and other assets
│   ├── styles.css             # CSS file for styling the web interface
│   └── temp/                  # Temporary storage for uploaded images
├── models/                    # Pre-trained models for digit recognition
│   └── my_model.h5            # Machine learning model (in Keras format) for predicting Sudoku digits
├── classification/            # Folder for system classification-related files
│   └── System_Sudoku_Solver   # Classification model or system logic for solving Sudoku
├── requirements.txt           # List of dependencies required to install and run the project
└── README.md                  # Documentation for the project
```


## Kontribusi
We'd love for you to contribute! If you want to help out, just fork this repository and create a pull request with your changes.

## License
This project is licensed under the MIT license. See the LICENSE file for more information.


## Teams
<div align="center">
  <table style="margin: auto;">
    <tr>
      <td align="center">
  <a href="https://github.com/rizkyyanuark">
    <img src="https://avatars.githubusercontent.com/u/82692777?v=4" width="100px;" alt="RizkyYanuarK"/>
  </a>
  <br />
  <sub>RizkyYanuarK</sub>
</td>
<td align="center">
  <a href="https://github.com/fadhilahnuria">
    <img src="https://avatars.githubusercontent.com/u/114966285?v=4" width="100px;" alt="Fadhilah Nuria Shinta"/>
  </a>
  <br />
  <sub>Fadhilah Nuria Shinta</sub>
</td>
<td align="center">
  <a href="https://github.com/prenji3">
    <img src="https://avatars.githubusercontent.com/u/171494212?v=4" width="100px;" alt="Rivadian Ardiansyah"/>
  </a>
  <br />
  <sub>Rivadian Ardiansyah</sub>
</td>
<td align="center">
  <a href="https://github.com/resharjuliand">
    <img src="https://avatars.githubusercontent.com/u/171216405?v=4" width="100px;" alt="Reshar Faldi Julianda"/>
  </a>
  <br />
  <sub>Reshar Faldi Julianda</sub>
</td>
  </table>
</div>
