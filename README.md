# Block Blast Solver (under construction)

This project automates the detection of "blank" and "non-blank" tiles from a grid-based visual interface, such as a game or image, using Python. It processes screenshots of the interface, divides them into grids, calculates the average color of each grid cell, and identifies "blank" cells using machine learning.

---

## Features

1. **Take Screenshot of Target Window:**
   - Captures a screenshot of a specific window.
   - Crops out borders and title bar for accurate grid processing.

2. **Grid-Based Image Processing:**
   - Divides an image into an 8x8 grid.
   - Calculates average color values for each grid cell.
   - Saves the average color data in a text file.

3. **Machine Learning Classifier:**
   - Uses a Random Forest Classifier to classify cells as "blank" or "non-blank" based on RGB values.

4. **Canvas Visualization:**
   - Creates a visual representation of the blank and non-blank cells in the form of a grid image.

5. **Automated Workflow:**
   - Automates the entire pipeline, from capturing a screenshot to generating a visual canvas.

6. **Flexible Integration:**
   - Outputs results as labeled text files and organized datasets for further use.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Libraries:
  - `pygetwindow`
  - `pyautogui`
  - `numpy`
  - `pandas`
  - `Pillow`
  - `matplotlib`
  - `scikit-learn`
  - `re`
  - `subprocess`

### Install Dependencies
```bash
pip install pygetwindow pyautogui numpy pandas pillow matplotlib scikit-learn
```

---

## How to Use

1. **Prepare the Environment:**
   - Ensure the target window (e.g., a game or app) is open and visible.

2. **Run the Script:**
   - Execute the script:
     ```bash
     python block_blast_detector.py
     ```

3. **Output Files:**
   - **`screenshot.png`**: Screenshot of the target window.
   - **`cropped_image.jpg`**: Cropped and processed screenshot.
   - **`average_colors.txt`**: RGB values and average colors of the grid.
   - **`average_colors_labeled.txt`**: Labeled text file with blank/non-blank statuses.
   - **`organized_colors.txt`**: Organized data with RGB values and statuses.
   - **`blockBlastCanvas.png`**: Visual representation of the blank/non-blank grid.

---

## Key Functions

### `take_content_screenshot`
Captures and saves a screenshot of the target window.

### `crop_image`
Crops the screenshot to the desired region.

### `divide_and_calculate_average_colors`
Divides the cropped image into an 8x8 grid and calculates average colors.

### `blankDetector`
Trains a Random Forest model to classify tiles as "blank" or "non-blank."

### `organize_colors`
Organizes color data into a structured dictionary and saves it to a file.

### `create_canvas`
Generates a grid visualization of blank and non-blank cells.

---

## Example Workflow

1. **Screenshot and Processing:**
   - The script captures a window screenshot, crops it, and processes it into an 8x8 grid.

2. **Classification:**
   - Each grid cell is classified as blank or non-blank using the trained model.

3. **Canvas Visualization:**
   - Outputs a visual grid (`blockBlastCanvas.png`) with:
     - **Red:** Non-blank cells.
     - **Black:** Blank cells.

---

## Notes

- Modify the `take_content_screenshot` function to match the title of the target window.
- The dataset used in `blankDetector` is synthetic. Train with real data for improved accuracy.
- Ensure the window is not minimized or obscured during the screenshot.

---

## Future Enhancements

1. Add support for dynamic grid sizes.
2. Train the classifier with diverse datasets for better generalization.
3. Enhance visualization with interactive or larger grids.
