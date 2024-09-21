from utils import preprocess, splitcells, CropCell, getOriginalNumbers, getBoxes, gridContour, centerPoints, displaySudoku, extract_sudoku_grid, sudokuGrid, predictNumbers, imPreprocess, predictNumbersCap, extractGrid, extractBox, gridContourCap
from sudoku_solver import solveSudoku
from tensorflow.keras.models import load_model
import cv2
import os
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
    img = cv2.resize(img, (450, 450))

    img2 = cv2.imread(f'{img_path}{img_name}')

    img_preprocess = preprocess(img)
    image_name_preprocess = changeName(img_name, "preprocess")
    cv2.imwrite(f"{img_path}{image_name_preprocess}", img_preprocess)
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

    img_boxes = img.copy()
    for box in getBoxes(img_boxes):
        cv2.rectangle(img_boxes,
                      tuple(box[0]),
                      tuple(box[1]),
                      (0, 255, 0), 2)
    image_name_gridline = changeName(img_name, "gridline")
    cv2.imwrite(f"{img_path}{image_name_gridline}", img_boxes)
    images.append(image_name_gridline)

    boxes = extractBox(img)

    numbers = predictNumbersCap(boxes)
    print(numbers)

    final_grid = sudokuGrid(numbers)
    print(final_grid)

    predicted_numbers_img = img.copy()
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
    print(solved_sudoku)

    solved_image = displaySudoku(
        img, solved_sudoku, original_numbers)
    image_name_solved = changeName(img_name, "solved")
    cv2.imwrite(f"{img_path}{image_name_solved}", solved_image)
    images.append(image_name_solved)

    return images, solved_sudoku
