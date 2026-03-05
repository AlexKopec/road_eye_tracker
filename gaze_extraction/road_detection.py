from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

TARGET_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign"
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #results = model(frame, classes=list(TARGET_CLASSES.keys()))  # just classifying traffic items
    results = model(frame)

    # This returns the frame with bounding boxes, labels, and confidence scores drawn
    annotated = results[0].plot()

    cv2.imshow("Dashcam Detector", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()