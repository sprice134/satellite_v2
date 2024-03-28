from roboflow import Roboflow
rf = Roboflow(api_key="qB79ns125fuIALjApoEe")
project = rf.workspace("particleresearch").project("demo-hoblh")
version = project.version(5)
dataset = version.download("yolov8")
