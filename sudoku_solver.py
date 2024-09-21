'''
Sudoku Solver
Here I use backtracking logic to solve this puzzle.
'''

# Libraries
import numpy as np

# Find if it is possible to put the number on that box


def getPossible(y, x, n, board):
    '''Try to find n on the row (y), column (x) and on the 3x3 square.
    x: Index in x
    y: Index in y
    n: Possible number in x and y position
    board: Matrix of 9x9 boxes
    '''
    # Loop through x row
    for i in range(0, 9):
        if board[i][x] == n:
            return False
    # Loop through y column
    for i in range(0, 9):
        if board[y][i] == n:
            return False
    # get the index of the 3x3 coordinates
    x_box = (x // 3) * 3
    y_box = (y // 3) * 3
    # Iterate the 3x3 box
    for i in range(0, 3):
        for j in range(0, 3):
            if board[y_box + i][x_box + j] == n:
                return False
    # Return True if the n number can be on the box
    return True

# Get base case for recursion in backtracking


def getBaseCase(board):
    '''Return True if there is no empty box on the board
    board: Matrix of 9x9 boxes
    '''
    # Loop the board from the end searching a blank box
    for row in range(8, -1, -1):
        for column in range(8, -1, -1):
            # If it find a 0 (blank box) return False
            if board[row][column] == 0:
                return False
    # If there are no white boxes return True and the Sudoku is complete
    return True

# Backtracking for solving sudoku


def solveSudoku(board):
    '''Solve the sudoku by checking every empty box.
    It is used backtracking to solve it.
    board: Matrix of 9x9 boxes
    '''
    # Loop through all rows
    for row in range(9):
        # Loop through all columns
        for column in range(9):
            # If there is an empty box
            if board[row][column] == 0:
                # Try to put a number between 1 and 9
                for number in range(1, 10):
                    if getPossible(row, column, number, board):
                        board[row][column] = number
                        # Call the function again for the next box
                        solveSudoku(board)
                        # Evaluate if the last empty box stills empty
                        # If it is not 0 it means that we COMPLETE the puzzle!
                        if getBaseCase(board) != 0:
                            return board
                        else:
                            # Assign a 0 to the returned box
                            # And continue looping between 1-9
                            board[row][column] = 0
                # Here we return when it is not possible
                # To put a number between 1-9
                # So it will continue to the box before
                return board
