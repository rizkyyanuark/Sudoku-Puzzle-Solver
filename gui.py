import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Load your pre-trained model
model = load_model('path_to_your_model.h5')

# Function to preprocess the image


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh

# Function to find the main contour of the Sudoku


def main_outline(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# Function to reframe the contour


def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 2), dtype=np.float32)

    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]

    return points_new

# Function to split the Sudoku image into 81 cells


def splitcells(img):
    rows = np.vsplit(img, 9)
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)
    return cells

# Function to crop cells


def CropCell(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
    return Cells_croped

# Function to read cells and predict digits


def read_cells(cell, model):
    result = []
    for image in cell:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        if probabilityValue > 0.65:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

# Sudoku solver functions


def next_box(quiz):
    for row in range(9):
        for col in range(9):
            if quiz[row][col] == 0:
                return (row, col)
    return False


def possible(quiz, row, col, n):
    for i in range(0, 9):
        if quiz[row][i] == n and row != i:
            return False
    for i in range(0, 9):
        if quiz[i][col] == n and col != i:
            return False
    row0 = (row) // 3
    col0 = (col) // 3
    for i in range(row0 * 3, row0 * 3 + 3):
        for j in range(col0 * 3, col0 * 3 + 3):
            if quiz[i][j] == n and (i, j) != (row, col):
                return False
    return True


def solve(quiz):
    val = next_box(quiz)
    if val is False:
        return True
    else:
        row, col = val
        for n in range(1, 10):
            if possible(quiz, row, col, n):
                quiz[row][col] = n
                if solve(quiz):
                    return True
                else:
                    quiz[row][col] = 0
        return


def Solved(quiz):
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("....................")
        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")
            if col == 8:
                print(quiz[row][col])
            else:
                print(str(quiz[row][col]) + " ", end="")

# GUI setup


class SudokuSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.geometry("800x600")

        self.video_label = Label(root)
        self.video_label.pack()

        self.capture_button = Button(
            root, text="Capture Image", command=self.capture_image)
        self.capture_button.pack()

        self.solve_button = Button(
            root, text="Solve Sudoku", command=self.solve_sudoku)
        self.solve_button.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_image = frame
            cv2.imwrite("captured_image.jpg", frame)
            print("Image Captured")

    def solve_sudoku(self):
        if hasattr(self, 'captured_image'):
            puzzle = cv2.resize(self.captured_image, (450, 450))
            su_puzzle = preprocess(puzzle)
            su_contour, hierarchy = cv2.findContours(
                su_puzzle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            su_biggest, su_maxArea = main_outline(su_contour)
            if su_biggest.size != 0:
                su_biggest = reframe(su_biggest)
                su_pts1 = np.float32(su_biggest)
                su_pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
                su_matrix = cv2.getPerspectiveTransform(su_pts1, su_pts2)
                su_imagewrap = cv2.warpPerspective(
                    puzzle, su_matrix, (450, 450))
                su_imagewrap = cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)
                sudoku_cell = splitcells(su_imagewrap)
                sudoku_cell_croped = CropCell(sudoku_cell)
                grid = read_cells(sudoku_cell_croped, model)
                grid = np.asarray(grid)
                grid = np.reshape(grid, (9, 9))
                if solve(grid):
                    Solved(grid)
                else:
                    print("Solution don't exist. Model misread digits.")
            else:
                print("No Sudoku puzzle detected")


if __name__ == "__main__":
    root = Tk()
    app = SudokuSolverApp(root)
    root.mainloop()
