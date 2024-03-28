from ultralytics import YOLO  # Import the YOLO class from the ultralytics package

# Initialize the model for segmentation
for i in ['n', 's', 'm', 'l', 'x']:
    model = YOLO(f'yolov8{i}-seg.pt')  
    model.train(data='demo.v5i.yolov8/data.yaml', epochs=500, project=f'models_{i}', patience = 50)  

    # Evaluate the model's performance on the validation set
    metrics = model.val()  
    print(metrics)