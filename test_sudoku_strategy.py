# ---------------------------
# INITIAL SUDOKU GRID (your puzzle)
# ---------------------------
initial = [
    [0, 0, 0, 9, 0, 0, 1, 0, 2],
    [7, 0, 0, 3, 0, 0, 6, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 3, 0],
    [9, 0, 0, 0, 0, 8, 7, 0, 0],
    [3, 0, 0, 0, 1, 0, 0, 0, 9],
    [0, 0, 6, 5, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 9, 0, 0, 6],
    [8, 0, 3, 0, 0, 6, 0, 0, 0],
]

# Make board a working copy
board = [row[:] for row in initial]


# ---------------------------
# STRATEGY FUNCTION
# ---------------------------
def strategy2(board, initial):
    def is_valid(row, col, number):
        # Check row and column
        for i in range(9):
            if board[row][i] == number or board[i][col] == number:
                return False
        # Check 3x3 box
        box_row, box_col = row - row % 3, col - col % 3
        for r in range(3):
            for c in range(3):
                if board[box_row + r][box_col + c] == number:
                    return False
        return True

    for row in range(9):
        for col in range(9):
            if initial[row][col] == 0 and board[row][col] == 0:
                for number in range(1, 10):
                    if is_valid(row, col, number):
                        return (row, col, number)
    return (-1, -1, -1)  # Backtrack: no solution found for this backtrack step

# CORRECTED STRATEGY BY CHATGPT
def strategy(board, initial):
    for row in range(9):
        for col in range(9):
            # Only consider cells that were empty in the initial puzzle
            # and are still empty now
            if initial[row][col] != 0 or board[row][col] != 0:
                continue

            for number in range(1, 10):
                # Check column
                col_ok = all(board[i][col] != number for i in range(9))
                # Check row
                row_ok = all(board[row][j] != number for j in range(9))

                # Check 3x3 subgrid
                subgrid_row = (row // 3) * 3
                subgrid_col = (col // 3) * 3
                subgrid_ok = all(
                    board[x][y] != number
                    for x in range(subgrid_row, subgrid_row + 3)
                    for y in range(subgrid_col, subgrid_col + 3)
                )

                if row_ok and col_ok and subgrid_ok:
                    return (row, col, number)

    # No valid move found (for the backtracking solver to interpret)
    return (-1, -1, -1)


def strategy(board, initial):
    for row in range(9):
        for col in range(9):
            if initial[row][col] == 0 and board[row][col] == 0:
                # Check row for empty cells
                row_fraud = {num for num in board[row] if 0 < num < 10}

                # Check column for empty cells
                col_fraud = {initial[r][col] for r in range(9) if 0 < initial[r][col] < 10}

                # Check 3x3 box for empty cells
                box_row_start = row // 3 * 3
                box_row_end = box_row_start + 3
                box_col_start = col // 3 * 3
                box_col_end = box_col_start + 3
                box_fraud = {board[r][c] for r in range(box_row_start, box_row_end)
                           for c in range(box_col_start, box_col_end)
                           if 0 < board[r][c] < 10 or (r != initial[box_row*3 + r//3][box_col*3 + c//3] and r != initial[box_row_row*3 + r] and r != initial[box_row*3 + r][box_col*3 + c//3] and r != initial[box_row*3 + r//3][box_col*3 + c//3
                           and col_c drunk folding not in inital 3x3 box]

                # Try placing numbers from 1-9
                for num in range(1, 10):
                    if num not in row_fraud and num not in col_fraud and num not in box_fraud:
                        return (row, col, num)
    return (-1, -1, -1)
# ---------------------------
# PRINT FUNCTION (optional)
# ---------------------------
def print_board(board):
    for r in range(9):
        row = ""
        for c in range(9):
            val = board[r][c]
            row += f"{val if val != 0 else '.'} "
            if c % 3 == 2 and c != 8:
                row += "| "
        print(row)
        if r % 3 == 2 and r != 8:
            print("-" * 21)
    print("\n")


# ---------------------------
# TEST THE STRATEGY
# ---------------------------
print("Starting board:")
print_board(board)

moves = 0

while True:
    r, c, n = strategy(board, initial)
    if r == -1:
        print("No further move possible.")
        break

    board[r][c] = n
    moves += 1
    print(f"Move {moves}: place {n} at ({r}, {c})")
    print_board(board)

print("Finished.")
