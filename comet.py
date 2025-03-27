import comet_ml

comet_ml.login(project_name="comet-example-yolo11-coco128")

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data="coco8.yaml",
    project="comet-example-yolo11-coco128",
    batch=32,
    save_period=1,
    save_json=True,
    epochs=3,
)