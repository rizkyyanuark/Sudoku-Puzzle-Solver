import cv2
import numpy as np
import os
from flask import url_for
from utils import extract_sudoku_grid, split_cells, crop_cells, read_cells, getOriginalNumbers, sudokuGrid
from tensorflow.keras.models import load_model
from sudoku_solver import solve_sudoku

# Path to the model
MODEL_PATH = 'models/my_model.h5'

# Load the model
model = load_model(MODEL_PATH)

# Function to display Sudoku with newly filled digits


def displaySudoku(image, solved_grid, original_numbers):
    img_copy = image.copy()  # Copy image to avoid modifying the original

    # Define grid parameters
    grid_size = image.shape[0] // 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Font size for digits
    font_thickness = 2  # Font thickness for better visibility
    color = (45, 113, 0)  # Green color for the newly filled numbers

    for i in range(9):
        for j in range(9):
            if [i, j] not in original_numbers and solved_grid[i][j] != 0:
                text = str(solved_grid[i][j])
                text_size = cv2.getTextSize(
                    text, font, font_scale, font_thickness)[0]
                text_x = (j * grid_size) + (grid_size - text_size[0]) // 2
                text_y = (i * grid_size) + (grid_size + text_size[1]) // 2
                cv2.putText(img_copy, text, (text_x, text_y), font,
                            font_scale, color, font_thickness)

    return img_copy


def process_image(filepath):
    # Read the image
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Failed to read the image from the given filepath.")

    # Folder uploads
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)

    # Extract the Sudoku grid
    sudoku_grid_image = extract_sudoku_grid(img)
    if sudoku_grid_image is None:
        raise ValueError("Failed to extract the Sudoku grid image.")

    # Ensure the extracted image has valid channels
    if len(sudoku_grid_image.shape) == 2:  # Grayscale image
        sudoku_grid_image = cv2.cvtColor(sudoku_grid_image, cv2.COLOR_GRAY2BGR)
    # 4-channel image
    elif len(sudoku_grid_image.shape) == 3 and sudoku_grid_image.shape[2] == 4:
        sudoku_grid_image = cv2.cvtColor(sudoku_grid_image, cv2.COLOR_BGRA2BGR)

    # Save the extracted Sudoku grid image
    sudoku_grid_image_path = os.path.join(uploads_dir, 'sudoku_grid.jpg')
    cv2.imwrite(sudoku_grid_image_path, sudoku_grid_image)

    # Split the grid into individual cells
    boxes = split_cells(sudoku_grid_image)
    if boxes is None:
        raise ValueError("Failed to split the Sudoku grid into cells.")

    # Crop the cells for better prediction
    cropped_cells = crop_cells(boxes)

    # Predict the numbers in each cell
    numbers = read_cells(cropped_cells, model)
    if numbers is None:
        raise ValueError("Failed to predict numbers from the cells.")

    # Create a 9x9 Sudoku grid
    sudoku_grid = sudokuGrid(numbers)
    if sudoku_grid is None or len(sudoku_grid) != 9 or any(len(row) != 9 for row in sudoku_grid):
        raise ValueError("Grid size must be 9x9")

    # Get the original numbers
    original_numbers = getOriginalNumbers(sudoku_grid)
    if original_numbers is None:
        raise ValueError(
            "Failed to get original numbers from the Sudoku grid.")

    # Solve the Sudoku
    if solve_sudoku(sudoku_grid):
        solved_grid = sudoku_grid
    else:
        raise ValueError("Failed to solve the Sudoku.")

    # Display the solved Sudoku on the original image
    solved_image = displaySudoku(
        sudoku_grid_image, solved_grid, original_numbers)
    if solved_image is None:
        raise ValueError("Failed to display the solved Sudoku.")

    # Draw the original numbers on the cropped image
    if sudoku_grid_image is None:
        raise ValueError("sudoku_grid_image is None")
    if numbers is None or len(numbers) != 81:
        raise ValueError("numbers is None or does not have 81 elements")

    predicted_numbers_img = sudoku_grid_image.copy()
    # Ukuran setiap kotak (grid) pada gambar
    grid_size = sudoku_grid_image.shape[0] // 9

    # Hitung pusat setiap kotak dan letakkan angka di tengah
    for idx, number in enumerate(numbers):
        if number != 0:  # Jika prediksi bukan 0, gambar angka
            # Mendapatkan baris dan kolom dari indeks
            row, col = divmod(idx, 9)
            # Titik tengah pada sumbu X
            center_x = (col * grid_size) + grid_size // 2
            # Titik tengah pada sumbu Y
            center_y = (row * grid_size) + grid_size // 2

            text = str(number)
            font = cv2.FONT_HERSHEY_COMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            # Menggeser teks agar tepat di tengah secara horizontal
            text_x = center_x - text_size[0] // 2
            # Menggeser teks agar tepat di tengah secara vertikal
            text_y = center_y + text_size[1] // 2

            # Gambar angka di tengah kotak
            cv2.putText(predicted_numbers_img, text,
                        (text_x, text_y), font, 1, (0, 255, 0), 2)

    # Simpan gambar yang telah diberi prediksi angka
    predicted_numbers_image_path = os.path.join(
        uploads_dir, 'predicted_numbers.jpg')
    cv2.imwrite(predicted_numbers_image_path, predicted_numbers_img)

    # Save the processed image (solved Sudoku)
    processed_image_path = os.path.join(uploads_dir, 'solved_sudoku.jpg')
    cv2.imwrite(processed_image_path, solved_image)

    # Normalize paths for compatibility
    original_filename = os.path.basename(filepath).replace('\\', '/')
    sudoku_grid_filename = os.path.basename(
        sudoku_grid_image_path).replace('\\', '/')
    processed_filename = os.path.basename(
        processed_image_path).replace('\\', '/')
    predicted_numbers_filename = os.path.basename(
        predicted_numbers_image_path).replace('\\', '/')

    result = {
        'original': filepath,
        'processed': processed_image_path,
        'solution': solved_grid.tolist() if solved_grid is not None else [],
        'images': [
            url_for('uploaded_file', filename=original_filename),
            url_for('uploaded_file', filename=sudoku_grid_filename),
            url_for('uploaded_file', filename=processed_filename),
            url_for('uploaded_file', filename=predicted_numbers_filename)
        ]
    }

    return result
