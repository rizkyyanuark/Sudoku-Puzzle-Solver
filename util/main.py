from util.process import preprocess, splitcells, CropCell, getOriginalNumbers, getBoxes, gridContour, centerPoints, displaySudoku, extract_sudoku_grid, sudokuGrid, predictNumbers, imPreprocess, predictNumbersCap, extractGrid, extractBox, gridContourCap
from util.sudoku_solver import solveSudoku
from tensorflow.keras.models import load_model
import cv2
import re
import numpy as np

MODEL_PATH = 'models/my_model.h5'
img_path = 'static/temp/'
model = load_model(MODEL_PATH)


def changeName(name, value):
    if matches := re.search(r"\w+", name):
        return f"{matches.group()}_{value}.png"


def process_image(img_name):
    images = []
    img = cv2.imread(f'{img_path}{img_name}')

    # Tambahkan pengecekan setelah membaca gambar
    if img is None:
        print(
            f"Error: Gambar {img_name} tidak ditemukan atau tidak dapat dibaca.")
        return [], None

    img = cv2.resize(img, (450, 450))

    img2 = cv2.imread(f'{img_path}{img_name}')
    if img2 is None:
        print(
            f"Error: Gambar {img_name} tidak ditemukan atau tidak dapat dibaca.")
        return [], None

    img_preprocess = preprocess(img)
    img_preprocess2 = preprocess(img2)
    image_name_preprocess = changeName(img_name, "preprocess")
    cv2.imwrite(f"{img_path}{image_name_preprocess}", img_preprocess2)
    images.append(image_name_preprocess)

    sudoku_grid_image = extract_sudoku_grid(img2)
    image_name_transperspec = changeName(img_name, "Transform Perspective")
    cv2.imwrite(f"{img_path}{image_name_transperspec}", sudoku_grid_image)
    images.append(image_name_transperspec)

    img_contour = gridContour(img_preprocess, img)

    img_boxes = sudoku_grid_image.copy()
    for box in getBoxes(img_boxes):
        cv2.rectangle(img_boxes,
                      tuple(box[0]),
                      tuple(box[1]),
                      (0, 255, 0), 2)
    image_name_gridline = changeName(img_name, "gridline")
    cv2.imwrite(f"{img_path}{image_name_gridline}", img_boxes)
    images.append(image_name_gridline)

    sudoku_cell = splitcells(img_contour)
    sudoku_cell_croped = CropCell(sudoku_cell)

    numbers = predictNumbers(sudoku_cell_croped, model)

    final_grid = sudokuGrid(numbers)

    predicted_numbers_img = sudoku_grid_image.copy()
    center_points = centerPoints(img_boxes)
    counter = 0
    for number in numbers:
        if number != 0:
            cv2.putText(predicted_numbers_img,
                        str(number),
                        center_points[counter],
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2)
        counter += 1

    image_name_prediction = changeName(img_name, "prediction")
    cv2.imwrite(f"{img_path}{image_name_prediction}", predicted_numbers_img)
    images.append(image_name_prediction)

    original_numbers = getOriginalNumbers(final_grid)

    solved_sudoku = solveSudoku(final_grid)

    solved_image = displaySudoku(
        sudoku_grid_image, solved_sudoku, original_numbers)
    image_name_solved = changeName(img_name, "solved")
    cv2.imwrite(f"{img_path}{image_name_solved}", solved_image)
    images.append(image_name_solved)

    return images, solved_sudoku


def process_image_cap(img_name):
    images = []
    img = cv2.imread(f'{img_path}{img_name}')

    # Tambahkan pengecekan setelah membaca gambar
    if img is None:
        print(
            f"Error: Gambar {img_name} tidak ditemukan atau tidak dapat dibaca.")
        return [], None

    img_preprocess = imPreprocess(img)
    image_name_preprocess = changeName(img_name, "preprocess")
    cv2.imwrite(f"{img_path}{image_name_preprocess}", img_preprocess)
    images.append(image_name_preprocess)

    sudoku_grid_image = extract_sudoku_grid(img)
    image_name_transperspec = changeName(img_name, "Transform Perspective")
    cv2.imwrite(f"{img_path}{image_name_transperspec}", sudoku_grid_image)
    images.append(image_name_transperspec)

    contour = gridContourCap(img_preprocess)

    # Convert to BGR for colored drawing

    img = extractGrid(img, contour)

    img_boxes = sudoku_grid_image.copy()
    for box in getBoxes(img_boxes):
        cv2.rectangle(img_boxes,
                      tuple(box[0]),
                      tuple(box[1]),
                      (0, 255, 0), 2)
    image_name_gridline = changeName(img_name, "gridline")
    cv2.imwrite(f"{img_path}{image_name_gridline}", img_boxes)
    images.append(image_name_gridline)

    boxes = extractBox(img)

    numbers1 = predictNumbersCap(boxes)
    print(numbers1)

    final_grid1 = sudokuGrid(numbers1)
    print(final_grid1)

    predicted_numbers_img1 = sudoku_grid_image.copy()
    center_points = centerPoints(img_boxes)
    counter = 0
    for number in numbers1:
        if number != 0:
            cv2.putText(predicted_numbers_img1,
                        str(number),
                        center_points[counter],
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2)
        counter += 1

    image_name_prediction = changeName(img_name, "prediction")
    cv2.imwrite(f"{img_path}{image_name_prediction}", predicted_numbers_img1)
    images.append(image_name_prediction)

    original_numbers1 = getOriginalNumbers(final_grid1)

    solved_sudoku1 = solveSudoku(final_grid1)
    print(solved_sudoku1)

    solved_image = displaySudoku(
        sudoku_grid_image, solved_sudoku1, original_numbers1)
    image_name_solved = changeName(img_name, "solved")
    cv2.imwrite(f"{img_path}{image_name_solved}", solved_image)
    images.append(image_name_solved)

    return images, solved_sudoku1
