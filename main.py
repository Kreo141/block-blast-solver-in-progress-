import pygetwindow as gw
import pyautogui
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import ast
import subprocess
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



def take_content_screenshot(window_title, save_path, border_size=8, title_bar_height=30):
    
    windows = gw.getAllTitles()
    
    if window_title not in windows:
        print(f"Window titled '{window_title}' not found.")
        return
    
    
    window = gw.getWindowsWithTitle(window_title)[0]
    
    
    if window.isMinimized:
        window.restore()
    
    
    window.activate()
    
    
    left, top, right, bottom = window.left, window.top, window.right, window.bottom
    
    
    left += border_size
    top += title_bar_height
    right -= border_size
    bottom -= border_size
    
    
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    
    
    screenshot.save(save_path)
    print(f"Screenshot saved to {save_path}")



def crop_image(image_path, top_left, bottom_right, save_path):
    
    image = Image.open(image_path)
    
    
    left, upper = top_left
    right, lower = bottom_right
    
    
    cropped_image = image.crop((left, upper, right, lower))
    
    
    cropped_image.save(save_path)
    print(f"Cropped image saved to {save_path}")



def calculate_average_color(image, left, upper, right, lower):
    
    tile = image.crop((left, upper, right, lower))
    
    
    tile_array = np.array(tile)
    
    
    average_color = tile_array.mean(axis=(0, 1))  
    return tuple(average_color.astype(int))  



def divide_and_calculate_average_colors(image_path, grid_size=8, save_path="averages.txt", marked_image_path="marked_image.jpg"):
    
    image = Image.open(image_path)
    
    
    image_width, image_height = image.size
    
    
    tile_width = image_width // grid_size
    tile_height = image_height // grid_size
    
    
    remaining_width = image_width % grid_size
    remaining_height = image_height % grid_size

    
    average_colors = []

    
    draw = ImageDraw.Draw(image)
    grid_color = (255, 0, 0)  
    line_thickness = 2  
    
    
    for row in range(grid_size):
        for col in range(grid_size):
            
            left = col * tile_width
            upper = row * tile_height
            
            
            right = left + tile_width + (remaining_width if col == grid_size - 1 else 0)
            lower = upper + tile_height + (remaining_height if row == grid_size - 1 else 0)
            
            
            avg_color = calculate_average_color(image, left, upper, right, lower)
            average_colors.append(avg_color)

            
            draw.rectangle([left, upper, right, lower], outline=grid_color, width=line_thickness)

            
            tile_number = row * grid_size + col + 1  
            print(f"Tile {tile_number} - Average color: {avg_color}")

    
    image.save(marked_image_path)
    print(f"Marked image saved to {marked_image_path}")

    
    with open(save_path, "w") as f:
        for idx, color in enumerate(average_colors):
            tile_number = idx + 1  
            f.write(f"Tile {tile_number} - Average color: {color}\n")



def blankDetector():
    data = {
        'R': [27, 219, 219, 36, 63, 33, 33, 32, 33, 224, 72, 38, 219, 47, 69, 39, 35, 41, 222, 42, 140, 142, 143, 134, 220, 141, 141, 40, 138, 138, 218, 50, 213, 66, 71, 39, 65, 46, 212, 74, 41, 47, 213, 41, 39, 213, 213, 78, 127, 133, 39, 39, 49, 201, 48, 199, 36, 42, 42, 42, 130, 45, 41, 121, 215, 140, 34, 215, 38, 33, 33, 126, 140, 41, 143, 41, 39, 215, 49, 65, 63, 221, 43, 217, 44, 69, 39, 38, 62, 71, 217, 41, 43, 210, 48, 39, 67, 216, 46, 42, 141, 49, 39, 67, 209, 48, 135, 69, 218, 212, 48, 201, 46, 144, 45, 62, 214, 51, 39, 190, 206, 210, 65, 42, 130, 67, 42, 41],
        'G': [35, 161, 160, 35, 93, 36, 35, 35, 35, 159, 93, 35, 155, 37, 91, 36, 36, 34, 156, 35, 86, 86, 87, 81, 160, 88, 87, 34, 86, 84, 152, 42, 154, 88, 94, 34, 87, 40, 151, 91, 40, 39, 148, 35, 34, 151, 148, 93, 82, 80, 34, 34, 39, 141, 39, 138, 39, 38, 38, 38, 81, 38, 38, 76, 117, 89, 35, 114, 36, 35, 35, 82, 90, 33, 89, 33, 34, 111, 37, 86, 94, 113, 35, 153, 37, 90, 36, 34, 94, 93, 153, 35, 36, 107, 37, 36, 94, 109, 37, 38, 86, 39, 34, 87, 108, 39, 82, 91, 112, 105, 37, 141, 44, 90, 41, 83, 109, 43, 34, 96, 151, 148, 86, 38, 81, 85, 40, 38],
        'B': [70, 51, 50, 65, 200, 73, 68, 68, 67, 49, 200, 69, 51, 63, 196, 74, 66, 66, 50, 63, 182, 185, 183, 176, 49, 185, 188, 67, 184, 182, 56, 65, 48, 195, 198, 69, 193, 72, 50, 183, 75, 73, 46, 63, 65, 48, 45, 177, 180, 177, 66, 65, 75, 46, 62, 53, 72, 70, 70, 71, 176, 74, 70, 167, 42, 188, 68, 41, 64, 67, 67, 174, 190, 66, 186, 66, 66, 41, 62, 189, 207, 40, 62, 48, 62, 194, 72, 66, 207, 201, 48, 63, 69, 38, 62, 69, 201, 37, 69, 71, 179, 70, 66, 183, 36, 73, 178, 201, 42, 33, 62, 47, 69, 177, 77, 191, 48, 73, 66, 35, 51, 49, 193, 74, 176, 193, 77, 71],
        'Label': [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
    }

    
    df = pd.DataFrame(data)

    
    X = df[['R', 'G', 'B']]

    
    y = df['Label']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    
    def process_input_file(input_filename, output_filename):
        
        with open(input_filename, 'r') as file:
            lines = file.readlines()

        
        output_lines = []

        
        pattern = r'Tile \d+ - Average color: \((\d+), (\d+), (\d+)\)'

        for line in lines:
            match = re.search(pattern, line)
            if match:
                r, g, b = map(int, match.groups())
                rgb_values = [[r, g, b]]
                prediction = model.predict(rgb_values)

                
                if prediction[0] == 0:
                    output_line = f"{line.strip()} - blank\n"
                else:
                    output_line = f"{line.strip()}\n"

                output_lines.append(output_line)

        
        with open(output_filename, 'w') as file:
            file.writelines(output_lines)

    
    process_input_file('average_colors.txt', 'average_colors_labeled.txt')



def organize_colors(input_file, output_file):
    
    r_values = []
    g_values = []
    b_values = []
    blank_status = []

    
    with open(input_file, 'r') as file:
        for line in file:
            
            match = re.search(r'Average color: \((\d+), (\d+), (\d+)\)', line)
            if match:
                r, g, b = map(int, match.groups())
                r_values.append(r)
                g_values.append(g)
                b_values.append(b)
                
                
                if 'blank' in line:
                    blank_status.append(0)  
                else:
                    blank_status.append(1)  

    
    data = {
        'R': r_values,
        'G': g_values,
        'B': b_values,
        'blank': blank_status
    }

    
    with open(output_file, 'w') as out_file:
        out_file.write("data = {\n")
        out_file.write(f"\t 'R': {r_values},\n")
        out_file.write(f"\t 'G': {g_values},\n")
        out_file.write(f"\t 'B': {b_values},\n")
        out_file.write(f"\t 'blank': {blank_status}\n")
        out_file.write("}\n")

    print(len(data['blank']))

    return data



def create_canvas(data):
    
    blank_status = data['blank']
    
    
    canvas = np.zeros((8, 8), dtype=int)
    
    
    for i in range(8):
        for j in range(8):
            index = i * 8 + j  
            
            if blank_status[index] == 1:
                canvas[i, j] = 1  
    
    
    cmap = plt.cm.colors.ListedColormap(['black', 'red'])
    
    
    plt.imshow(canvas, cmap=cmap)
    plt.axis('off')  
    
    
    plt.savefig('blockBlastCanvas.png', bbox_inches='tight', pad_inches=0)
    
    
    plt.show()

    
    take_content_screenshot("CPH2059", "screenshot.png")
    
    crop_image("screenshot.png", (20, 188), (330, 500), "cropped_image.jpg")
    
    divide_and_calculate_average_colors("cropped_image.jpg", grid_size=8, save_path="average_colors.txt")

    blankDetector()

    data = organize_colors('average_colors_labeled.txt', 'organized_colors.txt')

    create_canvas(data)

    dataAsString = ",".join(map(str, data['blank']))

    
    subprocess.run(["python", "nadeEngineTest.py", dataAsString])

