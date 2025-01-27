<div align="center">
  <h1> Sudoku Solver : Digital Image Processing and CNN</h1>
  <img src="https://github.com/rizkyyanuark/SudokuSolver-DataCitra/blob/main/util/sudoku.png" align="center" alt="Logo" width="275" height="275">
</div>

## 📝 Overview
This project is a web app that lets users upload an image of a Sudoku board or use a camera to capture one, then automatically solve the puzzle. It uses image processing and machine learning to detect and recognize the numbers, as well as a backtracking algorithm to solve the puzzle.

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
- **Add OCR Model** : EasyOCR

## 🚀 Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/rizkyyanuark/Sudoku-Puzzle-Solver.git
   cd Sudoku-Puzzle-Solver
2. **Create and activate a virtual environment:**
   - On Windows:
     ```
     python -m venv sudoku_venv
     sudoku_venv\Scripts\activate
     ```
   - on Macos/Linux:
     ```
     python3 -m venv sudoku_venv
     source sudoku_venv/bin/activate
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
   - Open your web browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to use the app.


## ⚙️ Project Structure
```
🗂️ Sudoku-Puzzle-Solver/
├── 📄 app.py                     
├── 📄 main.py                    
├── 📄 sudoku_solver.py           
├── 📄 utils.py                   
├── 📁 templates/                 
│   └── 📄 index.html             
├── 📁 static/                    
│   ├── 📄 styles.css             
│   └── 📁 temp/                  
├── 📁 models/                    
│   └── 📄 my_model.h5            
├── 📁 classification/            
│   └── 📄 System_Sudoku_Solver   
├── 📄 requirements.txt           
└── 📄 README.md
```

## 🎥 Demo Video

Check out the demo video to see the Sudoku Solver

<div align="center">
  <a href="https://www.linkedin.com/posts/rizkyyanuark_computervision-deeplearning-sudokusolver-activity-7245997572933255168-1HKo?utm_source=share&utm_medium=member_desktop" target="_blank">
    Watch Full Demo on LinkedIn
  </a>
  <br/>
  <a href="https://www.linkedin.com/posts/rizkyyanuark_computervision-deeplearning-sudokusolver-activity-7245997572933255168-1HKo?utm_source=share&utm_medium=member_desktop" target="_blank">
    <img src="https://github.com/rizkyyanuark/Sudoku-Puzzle-Solver/blob/ed15bef11268f939ff16cf001945e4bb48ae983e/material/demo.gif" alt="Report Video" width="600"/>
  </a>
</div>


## 🛠️ Contribution

We'd love for you to contribute! If you'd like to help out, feel free to fork this repository, make your changes, and submit a pull request. Let's make this project even better together!


## 📜 License

This project is licensed under the MIT license. See the [LICENSE](./LICENSE) file for more information.


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
  </table>
</div>
