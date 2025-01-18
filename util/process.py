import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import easyocr

# Load your custom model
model = tf.keras.models.load_model('models/my_model.h5')

# Global Variable
DIGITS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']


def preprocess(img):
    '''Preprocess the image to enhance contrast and improve detection'''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    img_invert = cv2.bitwise_not(img_thresh)
    return img_invert

# 2. Main outline detection


def imPreprocess(img):
    '''Preprocess the image for a better image detection
    img: The image to preprocess, a class 'numpy.ndarray'
    '''
    # Change to black and white for a better reading of the image
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur image
    image = cv2.GaussianBlur(image, (9, 9), 0)
    # Find boundaries in the image with Threshold
    image = cv2.adaptiveThreshold(image,
                                  255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,
                                  11,
                                  2)
    # Invert the image
    image = cv2.bitwise_not(image)
    # Dilate the image to thick the font and contours
    # We need to create a kernel to dilate
    kernel = np.array([[0., 1., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.]],
                      np.uint8)
    image = cv2.dilate(image, kernel)
    return image


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


def predictNumbersCap(boxes):
    '''Predict the numbers for each box,
    if no digit it will append 0
    boxes: Image of every box of the sudoku
    '''
    final_numbers = []
    # Configure the reades wirh easyocr
    reader = easyocr.Reader(['ru', 'en'])

    for box in boxes:
        # Blur each of the images
        box = cv2.blur(box, (3, 3))
        # Read each box and predict the number
        number = reader.readtext(box,
                                 allowlist='123456789',
                                 text_threshold=0.5,
                                 mag_ratio=2,
                                 contrast_ths=0.8)

        # Append 0 if there is no prediction
        if not number or not number[0][1].isnumeric() or number[0][2] < 0.8:
            final_numbers.append(0)
            continue

        # Extract the number from the data structure
        number = int(number[0][1])

        # Check if the number is grater than 9
        if number > 9:
            # Offset the borders by 1 pixel until the number is less or equal to 9
            while True:
                # Make an offset from the borders of the image
                # Give a variable for the offset
                px = 1
                box = box[0 + px:box.shape[0] - px, 0 + px:box.shape[1] - px]
                # Read and predict the number again
                number = reader.readtext(box,
                                         allowlist='123456789',
                                         text_threshold=0.5,
                                         mag_ratio=2,
                                         contrast_ths=0.8)
                # If the output is more or less than 1 item in the list
                if len(number) != 1:
                    continue
                # If the number is 1,2,3,4,5,6,7,8,9 append and break
                if number[0][1] in DIGITS:
                    final_numbers.append(int(number[0][1]))
                    break
        else:
            # If the number is predicted append to the list
            final_numbers.append(number)
    # Return a list of numbers
    return final_numbers


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


def splitcells(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            # Reduce noise
            denoised_box = cv2.fastNlMeansDenoising(box, None, 30, 7, 21)
            # Convert to black and white
            _, bw_box = cv2.threshold(
                denoised_box, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            boxes.append(bw_box)
    return boxes

# 5. Crop each cell for better prediction


def CropCell(cells):
    Cells_croped = []
    for image in cells:

        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)

    return Cells_croped

# 6. Read and predict the value of each cell using a trained model


def getOriginalNumbers(board):
    '''Get the initial number indexes
    board: Matrix of 9x9 boxes
    '''
    indexes = np.zeros_like(board)
    for row in range(9):
        for column in range(9):
            if board[row][column] != 0:
                indexes[row][column] = 1
    return indexes

# Center point of every box


def centerPoints(img):
    '''Get the center posotion of each box
    img: a rectangle image
    '''
    # Create a list with all the center point of each box
    center_points = []
    # Get the boxes and the coordinates to put the number
    for box in getBoxes(img):
        # First we get the center of the box
        # Then we translate it to a quarter (botton left corner)
        # To have the number in the center
        x = ((box[0][0]+box[1][0]) // 2) - ((box[1][0]-box[0][0])//4)
        y = ((box[0][1] + box[1][1]) // 2) + ((box[1][1]-box[0][1])//4)
        center_points.append(tuple([x, y]))
    return center_points


def gridContour(img, ori):
    '''Get the grid contours of the Sudoku image'''
    su_contour_1 = img.copy()
    su_contour_2 = img.copy()
    su_contour, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(su_contour_1, su_contour, -1, (0, 255, 0), 3)

    black_img = np.zeros((450, 450, 3), np.uint8)
    su_biggest, su_maxArea = main_outline(su_contour)
    if su_biggest.size != 0:
        su_biggest = reframe(su_biggest)
        cv2.drawContours(su_contour_2, su_biggest, -1, (0, 255, 0), 10)
        su_pts1 = np.float32(su_biggest)
        su_pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        su_matrix = cv2.getPerspectiveTransform(su_pts1, su_pts2)
        su_imagewrap = cv2.warpPerspective(ori, su_matrix, (450, 450))
        su_imagewrap = cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)
    return su_imagewrap


def gridContourCap(img):
    '''Detect the contour of the grid
    img: Preprocessed image, a class 'numpy.ndarray'
    '''
    # Detect all contours
    contours, h = cv2.findContours(img,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Rearrange the contours depending on each area
    # The biggest area first (the grid)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    grid = contours[0]
    return grid


def read_cells(cells, model):
    '''Predict numbers in Sudoku cells using a trained model and cross-check with EasyOCR'''
    result = []
    for image in cells:
        # preprocess the image as it was in the model
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = img.reshape(1, 32, 32, 1)
        # getting predictions and setting the values if probabilities are above 65%

        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)

        if probabilityValue > 0.65:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


def predictNumbers(boxes, model):
    '''Predict the numbers for each box using EasyOCR and a trained model,
    with cross-check between model and OCR predictions.
    boxes: Image of every box of the sudoku
    model: Trained ML model for predicting digits
    '''
    final_numbers = []
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

    for box in boxes:
        # Ensure the box is a numpy array
        if not isinstance(box, np.ndarray):
            box = np.array(box)

        # Blur each of the images
        box = cv2.blur(box, (3, 3))

        # --------- Step 1: Predict with EasyOCR ---------
        ocr_result = reader.readtext(
            box, allowlist='123456789', text_threshold=0.5, mag_ratio=2, contrast_ths=0.8)

        if ocr_result and ocr_result[0][1].isnumeric() and ocr_result[0][2] > 0.8:
            ocr_prediction = int(ocr_result[0][1])
        else:
            ocr_prediction = 0  # No valid OCR prediction

        # --------- Step 2: Predict with Model ---------
        # Prepare the image for model prediction
        img_array = np.asarray(box)
        img_array = img_array[4:img_array.shape[0] -
                              4, 4:img_array.shape[1] - 4]  # Cropping
        img_array = cv2.resize(img_array, (32, 32))  # Resizing
        img_array = img_array / 255.0  # Normalize
        img_array = img_array.reshape(1, 32, 32, 1)  # Reshape for model input

        # Predict using the model
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=-1)
        probability = np.amax(predictions)

        if probability > 0.65:  # Threshold for valid model prediction
            model_prediction = class_index[0]
        else:
            model_prediction = 0  # No valid model prediction

        # --------- Step 3: Cross-Check Between OCR and Model ---------
        if model_prediction == ocr_prediction:
            # If both agree, take the result
            final_numbers.append(model_prediction)
        elif model_prediction != 0 and ocr_prediction == 0:
            # If only model is confident
            final_numbers.append(model_prediction)
        elif model_prediction == 0 and ocr_prediction != 0:
            final_numbers.append(ocr_prediction)  # If only OCR is confident
        else:
            final_numbers.append(0)  # If neither is confident, append 0

    return final_numbers


def getBoxes(img):
    '''Divide square image into 81 boxes
    img: Square image of 920x920 pixels
    '''
    # Get the image # of rows of the image
    side = img.shape[:1]
    # List of boxes
    squares = []
    # Divide the side into 9
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            # Top left corner of a box
            p1 = (int(i * side), int(j * side))
            # Bottom right corner
            p2 = (int((i + 1) * side), int((j + 1) * side))
            squares.append((p1, p2))
    # Return a list of of the coordinates of the boxes (pt1 to pt2)
    return squares


def displaySudoku(image, solved_grid, original_numbers):
    img_copy = image.copy()
    grid_size = image.shape[0] // 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    color = (45, 113, 0)

    for i in range(9):
        for j in range(9):
            # Only draw numbers that are not in the original grid and are solved by the algorithm
            if original_numbers[i][j] == 0 and solved_grid[i][j] != 0:
                text = str(solved_grid[i][j])
                text_size = cv2.getTextSize(
                    text, font, font_scale, font_thickness)[0]
                text_x = (j * grid_size) + (grid_size - text_size[0]) // 2
                text_y = (i * grid_size) + (grid_size + text_size[1]) // 2
                cv2.putText(img_copy, text, (text_x, text_y), font,
                            font_scale, color, font_thickness)

    return img_copy


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


def sudokuGrid(numbers):
    '''Create a 9x9 Sudoku grid with all the values
    numbers: List of 81 numbers
    '''
    # Return a list reshaped by a grid of 9x9
    return np.array(numbers).reshape(9, 9)


def getVertices(contour):
    '''Get the coordinates of the 4 vertices of the grid
    contour: Array of each point that conforms the contour
    '''
    # Find the 4 vertices of the grid
    vertices = cv2.approxPolyDP(contour, 50, True)
    # Return False if there are not 4 points
    if len(vertices) != 4:
        return False
    # Reorganize vertices with a list comprehension and sorted
    vertices = [vertex[0] for vertex in vertices]
    # Organize boundary vertices (min sum to max sum)
    vertices = sorted(vertices, key=sum)
    # Organize de 2nd and 3rd vertex
    if vertices[1][0] < vertices[2][0]:
        temp = vertices[1]
        vertices[1] = vertices[2]
        vertices[2] = temp
    return vertices


def extractGrid(img, contour):
    '''Extract the grid from the image with the 4 vertices
    img: Original image, a class 'numpy.ndarray'
    '''
    # Change the format of the list of vertices
    vertices = np.float32(getVertices(contour))
    # Create a new rectangle image
    vertices_2 = np.float32([[0, 0], [920, 0], [0, 920], [920, 920]])
    # Apply Perspective Transform Algorithm
    try:
        matrix = cv2.getPerspectiveTransform(vertices, vertices_2)
    except cv2.error:
        return False
    grid = cv2.warpPerspective(img, matrix, (920, 920))
    return grid


def getBoxes(img):
    '''Divide square image into 81 boxes
    img: Square image of 920x920 pixels
    '''
    # Get the image # of rows of the image
    side = img.shape[:1]
    # List of boxes
    squares = []
    # Divide the side into 9
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            # Top left corner of a box
            p1 = (int(i * side), int(j * side))
            # Bottom right corner
            p2 = (int((i + 1) * side), int((j + 1) * side))
            squares.append((p1, p2))
    # Return a list of of the coordinates of the boxes (pt1 to pt2)
    return squares

# 6. Extract every box of the grid


def extractBox(img):
    '''Prepare image and extract each box from the sudoku
    for a better digit recognition
    img: Square image of 920x920 pixels
    '''
    # Get a grid of boxes to split the image
    grid = getBoxes(img)

    # Preprocess the image
    # Dilate image
    kernel = np.array([[0., 1., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.]],
                      np.uint8)
    img = cv2.dilate(img, kernel)
    # Change to black and white
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert image
    img = cv2.bitwise_not(img)

    # Invert color
    # Create a list to append all the split images
    squares = []
    for square in grid:
        # Slides [::] to travel from y[0] to y[1]
        # Slides [::] to travel from x[0] to x[1]
        square = img[square[0][1]:square[1][1], square[0][0]:square[1][0]]
        squares.append(square)
    return squares
