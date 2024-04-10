import matplotlib.pyplot as plt  # Import matplotlib for plotting
from PIL import Image, ImageDraw  # For loading images
import numpy as np
from ultralytics import YOLO
from samDemo import show_mask, show_points, show_box, mask_to_polygon, generate_random_points_within_polygon, point_to_polygon_distance, find_optimal_points, polygon_to_binary_mask, expand_bbox_within_border, fractal_dimension, apply_mask_to_image
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2
from shapely.geometry import Polygon, Point
from itertools import combinations
import random
from skimage.draw import polygon
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
import math
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from pycocotools import mask as maskUtils
import skimage.draw

def create_coco_annotations_from_polygons(polygons, image_width, image_height, image_filename):
    """Creates COCO annotations for a list of polygons."""
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add image information
    image_id = 1
    coco_output["images"].append({
        "id": image_id,
        "width": int(image_width),  # Ensure width is an int
        "height": int(image_height),  # Ensure height is an int
        "file_name": image_filename
    })

    # Define category information
    coco_output["categories"].append({
        "id": 1,
        "name": "Powder",
        "supercategory": "Powder"
    })

    # Add annotations for each polygon
    for annotation_id, polygon in enumerate(polygons, start=1):
        # Flatten the polygon coordinates and ensure all values are serializable
        segmentation = [list(map(float, point)) for point in polygon]

        # Calculate the bounding box [x,y,width,height] from the polygon
        min_x, min_y = np.min(polygon, axis=0)
        max_x, max_y = np.max(polygon, axis=0)
        width, height = max_x - min_x, max_y - min_y
        bbox = [float(min_x), float(min_y), float(width), float(height)]

        # Calculate area and convert it to float to ensure JSON serialization
        area = float(0.5 * np.abs(np.dot(polygon[:,0], np.roll(polygon[:,1], 1)) - np.dot(polygon[:,1], np.roll(polygon[:,0], 1))))

        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # Assuming all are "Powder"
            "segmentation": [segmentation],  # Modify to fit COCO format
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        })

    return coco_output

def polygons_to_rle(polygons, width, height):
    segm = []
    for polygon in polygons:
        # Create a binary mask from the polygon
        mask = np.zeros((height, width), dtype=np.uint8)
        rr, cc = skimage.draw.polygon(np.array(polygon)[:, 1], np.array(polygon)[:, 0], shape=(height, width))
        mask[rr, cc] = 1
        
        # Encode the mask to RLE format
        rle = maskUtils.encode(np.asfortranarray(mask))
        segm.append(rle)
    return segm



def convert_yolo_to_coco(images_dir, labels_dir, output_json_path):
    # Initialize the COCO dataset structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "Powder"}]
    }
    
    annotation_id = 1  # Unique ID for each annotation
    image_id = 1  # Unique ID for each image
    
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Load the image to get its dimensions
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        # Add image information to COCO dataset
        coco_data["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename,
        })
        
        # Corresponding label file
        label_file = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    values = line.split()
                    if len(values) < 6:
                        print(f"Ignore line in {label_path}: {line} does not contain enough values.")
                        continue
                    
                    class_id = int(values[0])  # Assuming always 0 for circle
                    points = list(map(float, values[1:]))
                    
                    # Normalize coordinates to image size
                    normalized_points = [(points[i] * width, points[i + 1] * height) for i in range(0, len(points)//2)]
                    
                    # Convert polygon to bounding box
                    x_values = [point[0] for point in normalized_points]
                    y_values = [point[1] for point in normalized_points]
                    x_min = min(x_values)
                    y_min = min(y_values)
                    x_max = max(x_values)
                    y_max = max(y_values)
                    
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, width, height]
                    
                    # Convert normalized polygon points to COCO segmentation format
                    segmentation = [[point[0], point[1]] for point in normalized_points]
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,  # Assuming only one class type (circle)
                        "bbox": bbox,
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                        "segmentation": segmentation  # Add segmentation data
                    })
                    
                    annotation_id += 1
        
        image_id += 1

    # Save the COCO data to a file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

def calculate_mAP(ground_truth_file, pred_file):
    # Load ground truth COCO JSON file
    coco_gt = COCO(ground_truth_file)

    # Load predicted COCO JSON file
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    # Initialize COCO evaluation object
    coco_eval = COCOeval(coco_gt)

    # Iterate over each image in predictions
    for pred_image in pred_data['images']:
        image_id = pred_image['id']

        # Get ground truth annotations for current image
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        ground_truth_anns = coco_gt.loadAnns(ann_ids)

        # Get predicted annotations for current image
        pred_anns = [ann for ann in pred_data['annotations'] if ann['image_id'] == image_id]

        # Convert predicted polygons to RLE format
        for pred_ann in pred_anns:
            polygons = pred_ann['segmentation']
            width = pred_image['width']
            height = pred_image['height']
            pred_ann['segmentation'] = polygons_to_rle(polygons, width, height)

        # Evaluate predictions for the current image
        coco_eval.cocoDt = coco_gt.loadRes(pred_anns)
        coco_eval.evaluate()
        coco_eval.accumulate()

    # Calculate mAP
    coco_eval.summarize()
    mAP = coco_eval.stats[0]  # mAP at IoU=0.50:0.95

    return mAP

def save_yolo_annotations(predictions, output_dir):
    """
    Save YOLO model predictions as annotations in YOLO TXT format.
    
    Args:
    - predictions (list): List of tuples, each containing predictions for one image.
                          Each tuple should contain the following information:
                          (image_filename, class_id, x_center, y_center, width, height)
    - output_dir (str): Directory path where the annotation files will be saved.
    
    Returns:
    - None
    """
    for prediction in predictions:
        image_filename, class_id, x_center, y_center, width, height = prediction
        annotation_filename = image_filename.replace(".jpg", ".txt")
        annotation_filepath = os.path.join(output_dir, annotation_filename)
        
        with open(annotation_filepath, 'a') as f:
            line = f"{class_id} {x_center} {y_center} {width} {height}\n"
            f.write(line)

def convert_coordinates_to_yolov8_format(coordinates, image_width, image_height):
    # Normalize x and y coordinates
    normalized_xy_pairs = [0]  # Starting with class ID 0 for YOLOv8 format
    for coord in coordinates:
        normalized_x = coord[0] / image_width
        normalized_y = coord[1] / image_height
        normalized_xy_pairs.extend([normalized_x, normalized_y])

    return normalized_xy_pairs

def save_list_of_lists_to_file(list_of_lists, file_path):
    """
    Save a list of lists to a file, with each list on its own line and spaces as delimiters.

    Parameters:
    - list_of_lists: The list of lists to save.
    - file_path: The path to the file where the data will be saved.
    """
    with open(file_path, 'w') as file:
        for sublist in list_of_lists:
            # Convert each element of the sublist to a string and join with spaces
            line = ' '.join(map(str, convert_coordinates_to_yolov8_format(sublist, 1024, 768)))
            file.write(line + '\n')  # Write the line to the file and add a newline character






# images_dir = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images'
# labels_dir = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/labels'
# output_json_path = 'coco_ground_truth.json'
# convert_yolo_to_coco(images_dir, labels_dir, output_json_path)


# Load the trained model (replace 'yolov8n-seg.pt' with your model's weight file)
# model = YOLO('models/yolov8s_trained_weights.pt')  # Use the path to your trained weights
model = YOLO('/home/sprice/satellite_v2/particleTest/modelOutputs/models_n/train/weights/best.pt') 

# Copper
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/Cu-Ni-Powder_250x_2_SE_V1_png.rf.eec5f31cbe6f51d8aa6a574a01f1883c.jpg'
# Big Ones
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S02_03_SE1_1000X24_png.rf.61ceee7fe0a4f4ccabd61c1e71524baf.jpg'
# Demo Sample
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S05_02_SE1_300X59_png.rf.234bd1c35d0f3a635fd6164b651601f9.jpg'
# Real Big
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/RHA_00-45_500X07_png.rf.2e24ff0e093484de86e43a21ef7e62cb.jpg'
# Agglomerations
image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/valid/images/RHA_00-45_500X11_png.rf.a1d468233106b607347416e301a98df1.jpg'
image = Image.open(image_path)


# sam_checkpoint = "model/sam_vit_l_0b3195.pth"
# model_type = "vit_l"
# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)


listOfPolygons = []
listOfBoxes = []
results = model(image)
count = 0
image = image.convert("RGBA")
for i in results[0]:
    polygon_points = i.masks.xy[0]  # This is the array of points
    print(convert_coordinates_to_yolov8_format(i.masks.xy[0], image.width, image.height))
    if len(polygon_points) > 2:
        listOfPolygons.append(polygon_points)
        listOfBoxes.append(i.boxes.xyxy[0])
        

save_list_of_lists_to_file(listOfPolygons, 'RHA_00-45_500X11.txt')
image_filename = image_path.split('/')[-1]

coco_annotations = create_coco_annotations_from_polygons(listOfPolygons, image.width, image.height, image_filename)

with open('coco_pred.json', 'w') as f:
    json.dump(coco_annotations, f, indent=4)



# ground_truth_file = 'coco_ground_truth.json'
# pred_file = 'coco_pred.json'
# mAP = calculate_mAP(ground_truth_file, pred_file)
# print("Mean Average Precision (mAP):", mAP)
