#This scripts joins the views separated in different pngs into a multiview png

import os
import re
import cv2
import numpy as np

def read_images_in_order(folder_path, max_rows, max_cols):
    lf_images = {}

    # List all files in the folder
    file_names = sorted(os.listdir(folder_path))

    # Regular expression pattern to extract indices from filenames
    pattern = re.compile(r"lf_(\d+)__row_(\d+)__column_(\d+)")

    # Iterate over each file
    for file_name in file_names:
        # Match filename pattern
        match = pattern.match(file_name)
        if match:
            lf_index, row_index, col_index = map(int, match.groups())
            print(file_name, row_index, col_index)
            img = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_COLOR)
            lf_images.setdefault(lf_index, {})[(row_index, col_index)] = img
            

    # Concatenate images in order
    lf_concatenated = []
    for row_index in range(max_rows):
        row_images = []
        for col_index in range(max_cols):
            img = lf_images.get(lf_index, {}).get((row_index, col_index))
            if img is not None:
                row_images.append(img)
        if row_images:
            lf_concatenated.append(np.concatenate(row_images, axis=1))

    return np.concatenate(lf_concatenated, axis=0)

if __name__ == "__main__":
    folder_path = "/home/machado/Downloads/chessboard(1)/chessboard/images/0"
    max_rows = 15
    max_cols = 15
    lf_image = read_images_in_order(folder_path, max_rows, max_cols)

    # Save the concatenated LF image
    output_file_path = folder_path+"/all.png"
    cv2.imwrite(output_file_path, lf_image)

    print(f"Concatenated LF image saved as {output_file_path}")
