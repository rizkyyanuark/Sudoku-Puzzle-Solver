def is_valid(board, row, col, num):
    """Check if a number can be placed in a given position on the Sudoku board."""
    # Check if the number is already in the row
    for i in range(9):
        if board[row][i] == num:
            return False

    # Check if the number is already in the column
    for i in range(9):
        if board[i][col] == num:
            return False

    # Check if the number is already in the 3x3 sub-grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False

    return True


def solve_sudoku(board):
    """Solve the Sudoku puzzle using backtracking."""
    empty = find_empty_location(board)
    if not empty:
        return True  # Puzzle solved

    row, col = empty

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0  # Undo the move

    return False


def find_empty_location(board):
    """Find an empty location on the Sudoku board."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return (row, col)
    return None
