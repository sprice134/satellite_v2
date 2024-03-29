import matplotlib.pyplot as plt  # Import matplotlib for plotting
from PIL import Image, ImageDraw  # For loading images
import numpy as np
from ultralytics import YOLO
from samDemo import show_mask, show_points, show_box, mask_to_polygon, generate_random_points_within_polygon, point_to_polygon_distance, find_optimal_points, polygon_to_binary_mask, expand_bbox_within_border
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2
from shapely.geometry import Polygon, Point
from itertools import combinations
import random


# Load the trained model (replace 'yolov8n-seg.pt' with your model's weight file)
# model = YOLO('models/yolov8s_trained_weights.pt')  # Use the path to your trained weights
model = YOLO('/home/sprice/satellite_v2/particleTest/modelOutputs/models_n/train/weights/best.pt') 
image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S02_03_SE1_1000X24_png.rf.61ceee7fe0a4f4ccabd61c1e71524baf.jpg'
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S05_02_SE1_300X59_png.rf.234bd1c35d0f3a635fd6164b651601f9.jpg'
image = Image.open(image_path)


sam_checkpoint = "model/sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


listOfPolygons = []
listOfBoxes = []
results = model(image)
count = 0
image = image.convert("RGBA")
for i in results[0]:
    polygon_points = i.masks.xy[0]  # This is the array of points
    if len(polygon_points) > 2:
        listOfPolygons.append(polygon_points)
        listOfBoxes.append(i.boxes.xyxy[0])
        '''To Save, uncommnet:'''
        # tmp = Image.new('RGBA', image.size)
        # draw = ImageDraw.Draw(tmp)
        # draw.polygon(polygon_points, fill=(30, 144, 255, 180))  # Semi-transparent fill
        # # Composite the original image with the temporary image using alpha blending
        # image_with_polygon = Image.alpha_composite(image, tmp)
        # # Save the result to a file
        # image_with_polygon.convert('RGB').save(f'outputImages/yoloMask/{count}.png')
        # count += 1

INDEX = 17
poly = listOfPolygons[INDEX]
box = listOfBoxes[INDEX]
box = box.cpu().numpy()
box = np.array(expand_bbox_within_border(box[0], box[1], box[2], box[3], image.width, image.height, expansion_rate = 0.1))
mask = polygon_to_binary_mask(poly, image.height, image.width)
concave_polygon = Polygon(poly)
sampled_points = generate_random_points_within_polygon(concave_polygon, 50)
optimal_points = find_optimal_points(sampled_points, concave_polygon, num_result_points=3, border_weight=2)
optimal_points_xy = [[point.x, point.y] for point in optimal_points]
op_x, op_y = zip(*optimal_points_xy)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('off')
show_mask(mask, plt.gca())
plt.savefig(f'outputImages/yoloPipeline/yoloMask.png')
plt.plot(op_x, op_y, 'ro', markersize=5)
plt.savefig(f'outputImages/yoloPipeline/yoloCentralPointTest.png')



image = cv2.imread(image_path)
cv2.imwrite('outputImages/yoloPipeline/prePrediction.png', image)


predictor.set_image(image)
input_point = np.array(optimal_points_xy)
input_label = np.array([1, 1, 1])

# input_point = np.array([[600, 500], [500, 400], [600, 400]])
# input_label = np.array([1, 1, 1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=box[None, :],
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 0:
        print('Points found')
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_box(box, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f'outputImages/yoloPipeline/output_mask_{i+1}_centralPoints.png')