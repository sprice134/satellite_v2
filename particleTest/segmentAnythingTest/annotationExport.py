import matplotlib.pyplot as plt  # Import matplotlib for plotting
from PIL import Image, ImageDraw  # For loading images
import numpy as np
from ultralytics import YOLO
from samDemo import generate_random_points_within_polygon, find_optimal_points, polygon_to_binary_mask, expand_bbox_within_border
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
from shapely.geometry import Polygon
import cv2
import supervision as sv


def convert_coordinates_to_yolov8_format(coordinates, image_width, image_height):
    # Normalize x and y coordinates
    normalized_xy_pairs = [0]  # Starting with class ID 0 for YOLOv8 format
    for coord in coordinates:
        normalized_x = coord[0] / image_width
        normalized_y = coord[1] / image_height
        normalized_xy_pairs.extend([normalized_x, normalized_y])

    return normalized_xy_pairs

def save_list_of_lists_to_file(list_of_lists, file_path):
    with open(file_path, 'w') as file:
        for sublist in list_of_lists:
            # Convert each element of the sublist to a string and join with spaces
            line = ' '.join(map(str, convert_coordinates_to_yolov8_format(sublist, 1024, 768)))
            file.write(line + '\n')  # Write the line to the file and add a newline character


model = YOLO('/home/sprice/satellite_v2/particleTest/modelOutputs/models_n/train/weights/best.pt') 


# Poorly annotated training sample
# image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Rec Cu Ni Powder_250x_3_SE_V1.png'
# image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Rec Cu Ni Powder_250x_6_SE_V1.png'
# image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Rec Cu Ni Powder_250x_7_SE_V1.png'
# image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Rec Cu Ni Powder_250x_8_SE_V1.png'
# image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Cu Ni Powder_250x_7_SE_V1.png'
# image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Cu Ni Powder_250x_8_SE_V1.png'
# image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Cu Ni Powder_250x_9_SE_V1.png'
image_path = '/home/sprice/satellite_v2/Powder Characterization/4-11/Cu Ni Powder_250x_10_SE_V1.png'
image = Image.open(image_path)
filePath = image_path.split('/')[-1].split('.')[0] + '.txt'


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
    # print(convert_coordinates_to_yolov8_format(i.masks.xy[0], image.width, image.height))
    if len(polygon_points) > 2:
        listOfPolygons.append(polygon_points)
        listOfBoxes.append(i.boxes.xyxy[0])


samPolygons = []
for INDEX in range(len(listOfPolygons)):
    poly = listOfPolygons[INDEX]
    box = listOfBoxes[INDEX]
    box = box.cpu().numpy()
    # print(image)
    box = np.array(expand_bbox_within_border(box[0], box[1], box[2], box[3], image.width, image.height, expansion_rate = 0.1))
    mask = polygon_to_binary_mask(poly, image.height, image.width)
    concave_polygon = Polygon(poly)
    sampled_points = generate_random_points_within_polygon(concave_polygon, 50)
    optimal_points = find_optimal_points(sampled_points, concave_polygon, num_result_points=3, border_weight=2)
    optimal_points_xy = [[point.x, point.y] for point in optimal_points]
    op_x, op_y = zip(*optimal_points_xy)
    loop_image = cv2.imread(image_path)


    predictor.set_image(loop_image)
    input_point = np.array(optimal_points_xy)
    input_label = np.array([1, 1, 1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=box[None, :],
        multimask_output=True,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i == 0:
            samPolygons.append(sv.mask_to_polygons(mask)[0])

save_list_of_lists_to_file(samPolygons, filePath)