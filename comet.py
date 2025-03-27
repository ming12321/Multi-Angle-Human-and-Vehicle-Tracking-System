import comet_ml

comet_ml.login(project_name="multi_angle_human_and_vehicle_tracking_system")

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")

# Train the model
results = model.train(
    data="First_model\data.yaml",
    project="multi_angle_human_and_vehicle_tracking_system",
    batch=32,
    save_period=1,
    save_json=True,
    epochs=3,
    imgsz=640,
)