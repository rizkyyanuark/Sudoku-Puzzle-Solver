import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your custom model
model = tf.keras.models.load_model('models/my_model.h5')

# 1. Preprocess the image


def preprocess(img):
    '''Preprocess the image to enhance contrast and improve detection'''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    img_invert = cv2.bitwise_not(img_thresh)
    return img_invert

# 2. Main outline detection


def main_outline(contours):
    '''Get the largest square contour for the Sudoku grid'''
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

# 3. Reframe points for perspective correction


def reframe(points):
    '''Rearrange the points of the largest contour'''
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

# 4. Split cells of the Sudoku grid into individual boxes


def split_cells(img):
    '''Split the warped Sudoku grid into 81 individual cells'''
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            # Ensure the box is in grayscale
            if len(box.shape) == 3:
                box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            denoised_box = cv2.fastNlMeansDenoising(box, None, 30, 7, 21)
            _, bw_box = cv2.threshold(
                denoised_box, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            boxes.append(bw_box)
    return boxes

# 5. Crop each cell for better prediction


def crop_cells(cells):
    '''Crop unnecessary borders of each Sudoku cell'''
    cropped_cells = []
    for img in cells:
        cropped_img = np.array(img)[4:46, 6:46]
        cropped_cells.append(Image.fromarray(cropped_img))
    return cropped_cells

# 6. Read and predict the value of each cell using a trained model


def read_cells(cells, model):
    '''Predict numbers in Sudoku cells using a trained model'''
    results = []
    for img in cells:
        img_array = np.asarray(img)
        img_array = img_array[4:img_array.shape[0] -
                              4, 4:img_array.shape[1] - 4]
        img_array = cv2.resize(img_array, (32, 32))
        img_array = img_array / 255
        img_array = img_array.reshape(1, 32, 32, 1)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=-1)
        probability = np.amax(predictions)

        if probability > 0.65:
            results.append(class_index[0])
        else:
            results.append(0)  # If confidence is low, append 0
    return results

# 7. Warp perspective and extract Sudoku grid


def extract_sudoku_grid(image):
    '''Extract the Sudoku grid using contour detection and perspective transformation'''
    processed_img = preprocess(image)
    contours, _ = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest, max_area = main_outline(contours)
    if biggest.size == 0:
        return None  # If no grid is found

    biggest = reframe(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    wrapped_image = cv2.warpPerspective(image, matrix, (450, 450))
    return wrapped_image

# 8. Create a 9x9 Sudoku grid


def sudokuGrid(numbers):
    '''Create a 9x9 Sudoku grid with all the values
    numbers: List of 81 numbers
    '''
    # Return a list reshaped by a grid of 9x9
    return np.array(numbers).reshape(9, 9)


def getOriginalNumbers(board):
    '''Get the initial number indexes
    board: Matrix of 9x9 boxes
    '''
    indexes = []
    for row in range(9):
        for column in range(9):
            if board[row][column] != 0:
                indexes.append([row, column])
    return indexes


def centerPoints(boxes):
    '''Calculate the center points of each box in the grid'''
    center_points = []
    for box in boxes:
        h, w = box.shape[:2]
        center_points.append((w // 2, h // 2))
    return center_points


def preprocess_box(image):
    '''Preprocess the image to highlight the grid lines'''
    sudoku_a = cv2.resize(image, (450, 450))
    gray = cv2.cvtColor(sudoku_a, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 6)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)
    threshold_img = cv2.adaptiveThreshold(
        blur,
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return threshold_img


def getBoxes(img, save_path):
    '''Detect the grid and return the coordinates of each box'''
    processed_img = preprocess_box(img)

    # Finding the outline of the sudoku puzzle in the image
    contour_1 = img.copy()
    contours, hierarchy = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_1, contours, -1, (0, 255, 0), 3)

    # Save the contour detection result
    cv2.imwrite(save_path, contour_1)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust the area threshold as needed
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                # Extract the bounding box coordinates
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.8 < aspect_ratio < 1.2:  # Ensure the box is roughly square
                    boxes.append(((x, y), (x + w, y + h)))

    return boxes
