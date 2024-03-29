import matplotlib.pyplot as plt  # Import matplotlib for plotting
from PIL import Image, ImageDraw  # For loading images
import numpy as np
from ultralytics import YOLO

# Load the trained model (replace 'yolov8n-seg.pt' with your model's weight file)
# model = YOLO('models/yolov8s_trained_weights.pt')  # Use the path to your trained weights
model = YOLO('/home/sprice/satellite_v2/particleTest/models_x/train/weights/best.pt') 
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S05_02_SE1_300X59_png.rf.234bd1c35d0f3a635fd6164b651601f9.jpg'
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/valid/images/RHA_00-45_500X11_png.rf.a1d468233106b607347416e301a98df1.jpg'
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/valid/images/Sc1Tile_001-001-000_0-000_png.rf.eb8b0b9c2d61e13262bd82a4d1684140.jpg'
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/Rec-Cu-Ni-Powder_250x_3_SE_png.rf.354f7edbb16a86da549faafcbfa256a7.jpg'
image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S02_03_SE1_1000X24_png.rf.61ceee7fe0a4f4ccabd61c1e71524baf.jpg'

image = Image.open(image_path)

# Perform prediction (detection)
results = model(image)

annotated_image = results[0].plot(font_size=15, pil=True)


image2 = Image.open(image_path).convert('RGBA')

# Go through each result to draw the masks
for i in results[0]:
    # Extract the polygon points; convert them to the expected format
    # Each item in 'polygon_points' is expected to be a NumPy array of shape [n, 2]
    polygon_points = i.masks.xy[0]  # This is the array of points
    if len(polygon_points) != 0:
        # Convert the array of points to a flat list of coordinates
        flat_polygon_points = [coord for pair in polygon_points for coord in pair]

        # Create a blank (black) L-mode image for drawing the mask
        mask_img = Image.new('L', image2.size, 0)
        mask_draw = ImageDraw.Draw(mask_img)

        # Draw the polygon on the mask image using the extracted and flattened points
        mask_draw.polygon(flat_polygon_points, outline=1, fill=255)  # Fill with white

        # Convert the mask image to an RGBA image (so we can blend it)
        mask_rgba = Image.new('RGBA', mask_img.size)
        # Create a red mask (you can change the color)
        red_mask = Image.new('RGBA', mask_img.size, (255, 0, 0, 128))  # Red with 50% opacity
        # Paste the red mask using the white areas of mask_img as the mask
        mask_rgba.paste(red_mask, (0, 0), mask_img)

        # Overlay the red mask onto the grayscale image
        image2 = Image.alpha_composite(image2, mask_rgba)

# Convert back to RGB for display or saving
final_image = image2.convert('RGB')

# Display or save the final image
final_image.save('output2.png')
