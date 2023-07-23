import cv2
import math
import time
from ultralytics import YOLO
#model
model = YOLO("yolo-Weights/yolov8n.pt")
classNames = ["person"]
#webcam init
camera = cv2.VideoCapture(0)
camera.set(3, 50)
camera.set(4, 50)
camera2 = cv2.VideoCapture(1)
camera2.set(3, 50)
camera2.set(4, 50)

def resize_frame(frame, resolution):
    return cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)


while True:
    ret, frame= camera.read()
    ret2, frame2 = camera2.read()
    results = model(frame, stream=True)
    results2 = model(frame2, stream=True)
    if not (ret and ret2):
        break
    #webcam 1
    for r in results:
        boxes = r.boxes
        cnt = 0
        for box in boxes:
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)
            if confidence < 0.8:
                continue
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])
            if cls == 0:
                cnt+=1
        cv2.putText(frame,"people in room: "+str(cnt) , [5,20], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.imshow('Webcam 1', resize_frame(frame,(400,300)))
    #webcam2
    for r in results2:
        boxes = r.boxes
        cnt = 0
        for box in boxes:
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)
            if confidence < 0.8:
                continue
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # class name
            cls = int(box.cls[0])
           #print("Class name -->", classNames[cls])
            if cls == 0:
                cnt+=1
        cv2.putText(frame2,"people in room: "+str(cnt) , [5,20], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.imshow('Webcam 2', resize_frame(frame2,(400,300)))
    
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1)== 27: #27 is esc
        break
    time.sleep(0.3)

camera.release()
camera2.release()
cv2.destroyAllWindows()