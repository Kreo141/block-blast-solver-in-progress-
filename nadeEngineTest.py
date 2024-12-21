import tkinter as tk
import numpy as np
import sys

def can_fit(blank, block, start_row, start_col, grid_size):
    block_rows, block_cols = block.shape
    for i in range(block_rows):
        for j in range(block_cols):
            
            if block[i, j] == 1:
                
                if start_row + i >= grid_size or start_col + j >= grid_size or blank[start_row + i, start_col + j] == 1:
                    return False
    return True

def place_block(blank, block, start_row, start_col, grid_size):
    block_rows, block_cols = block.shape
    for i in range(block_rows):
        for j in range(block_cols):
            if block[i, j] == 1:
                blank[start_row + i, start_col + j] = 1
    return blank

def solve(blank, blocks, grid_size):
    blank = np.array(blank).reshape((grid_size, grid_size))
    
    
    placements = []
    
    
    for block in blocks:
        block = np.array(block)
        block_placed = False
        
        
        for i in range(grid_size):
            for j in range(grid_size):
                if can_fit(blank, block, i, j, grid_size):
                    blank = place_block(blank, block, i, j, grid_size)
                    placements.append((block, i, j))  
                    block_placed = True
                    break
            if block_placed:
                break
    
    return blank, placements


def draw_canvas(grid, grid_size, canvas_title, placements=None):
    window = tk.Tk()
    window.title(canvas_title)
    
    
    canvas = tk.Canvas(window, width=grid_size * 50, height=grid_size * 50)
    canvas.pack()

    
    for i in range(grid_size):
        for j in range(grid_size):
            color = 'red' if grid[i, j] == 1 else 'black'
            canvas.create_rectangle(j * 50, i * 50, (j + 1) * 50, (i + 1) * 50, fill=color, outline='gray')

    
    if placements:
        for block, start_row, start_col in placements:
            block_rows, block_cols = block.shape
            for i in range(block_rows):
                for j in range(block_cols):
                    if block[i, j] == 1:
                        canvas.create_rectangle((start_col + j) * 50, (start_row + i) * 50,
                                                (start_col + j + 1) * 50, (start_row + i + 1) * 50,
                                                fill='blue', outline='gray')

    window.mainloop()

blank = []
block = []

if len(sys.argv) > 1:
    blank = list(map(int, sys.argv[1].split(',')))
else:
    print("No argument received in (NO BLOCKS)")
    exit()




blocks = [
    [[1, 1], [1, 0]],  
    [[1, 1, 1], [1, 0, 0]],  
    [[0, 1], [1, 1], [1, 0]]   
]

grid_size = 8  


result, placements = solve(blank, blocks, grid_size)


blank_grid = np.array(blank).reshape((grid_size, grid_size))
draw_canvas(blank_grid, grid_size, "Before Placing Blocks")


draw_canvas(result, grid_size, "After Placing Blocks", placements)
