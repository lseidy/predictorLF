from PIL import Image

def extract_and_save_square_from_png(png_file_path, output_image_path):
    # Define the dimensions of the square cut
    cut_width, cut_height = 520, 464

    # Define the starting and ending coordinates for the square cut
    start_x, start_y = 4152, 520
    end_x, end_y = start_x + cut_width, start_y + cut_height

    # Open the PNG file using PIL
    image = Image.open(png_file_path)

    # Crop the square region
    image_cropped = image.crop((start_x, start_y, end_x, end_y))

    # Save the resulting image
    image_cropped.save(output_image_path)

# Example usage:
#png_file_path = '/home/idm/New_Extracted_Dataset/EPFL/Lenslet_8x8_RGB/Studio/Ankylosaurus_&_Diplodocus_1.png'
#output_image_path = 'head2_anky-1.png'
#extract_and_save_square_from_png(png_file_path, output_image_path)
#