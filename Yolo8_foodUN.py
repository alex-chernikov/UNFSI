#    copy this file to Yolo8 folder


def main():
    from ultralytics import YOLO
    model = YOLO('yolov8m.pt')
    results = model.train(data='food_un.yaml', epochs=10, imgsz=640,batch=48)
    print(results)
