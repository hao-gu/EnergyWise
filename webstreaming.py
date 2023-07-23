#from imutils.video import VideoStream
from flask import Response, Flask 
from flask import render_template
import torch
import threading
import argparse
import datetime
import imutils
import time
import cv2
import math
import time
from ultralytics import YOLO

outputframe = None
outputframe2 = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0).start()
#time.sleep(2.0)
#model
model = YOLO("yolo-Weights/yolov8n.pt")
#model.to("cuda")
classNames = ["person"]
#webcam init
camera = cv2.VideoCapture(0)
camera.set(3, 50)
camera.set(4, 50)
camera2 = cv2.VideoCapture(1)
camera2.set(3, 50)
camera2.set(4, 50)

@app.route("/")
def index():
	return render_template("index.html")
def resize_frame(frame, resolution):
    return cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
def detect_motion():
    print("detecing motion")
    global camera, camera2, outputframe, outputframe2, lock
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
        #cv2.imshow('Webcam 1', resize_frame(frame,(400,300)))
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
        #cv2.imshow('Webcam 2', resize_frame(frame2,(400,300)))
        frame =  resize_frame(frame,(400,300))
        frame2 =  resize_frame(frame2,(400,300))
        with lock:
            outputframe = frame.copy()
            outputframe2 = frame2.copy()  
        #if cv2.waitKey(1) == ord('q') or cv2.waitKey(1)== 27: #27 is esc
            #break
        time.sleep(0.3)
def generate():
	# grab global references to the output frame and lock variables
	global outputframe, outputframe2, lock
	print("HIHIHIHIHIHi")
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputframe is None:
				continue
			print("not none")
			(flag, encodedImage) = cv2.imencode(".jpg", outputframe)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		print("yielded")
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
def generate2():
	# grab global references to the output frame and lock variables
	global outputframe, outputframe2, lock
	print("HIHIHIHIHIHi")
	while True:
		#print('in the loospsppspsps')
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputframe2 is None:
				continue
			print("not none")
			(flag2, encodedImage2) = cv2.imencode(".jpg", outputframe2)
			# ensure the frame was successfully encoded
			if not flag2:
				continue
		print("yielded")
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage2) + b'\r\n')
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
@app.route("/video_feed2")
def video_feed2():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate2(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	'''ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())'''
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion)
	t.daemon = True
	t.start()
	# start the flask app
	app.run(debug=True, threaded=True, use_reloader=False)
	print("hi")
# release the video stream pointer
camera.release()
camera2.release()   