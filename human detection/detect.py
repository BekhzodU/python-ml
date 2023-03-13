import cv2
import numpy as np
import time

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

# open webcam video stream from mobile
cap = cv2.VideoCapture('http://192.168.100.11:8080/video')

last_capture_time = 0
capture_interval = 0.5 

while(True):
    current_time = time.time()
    #interval to capture frames every 0.5 seconds
    if (current_time - last_capture_time >= capture_interval):
        ret, frame = cap.read()
        last_capture_time = current_time

        # resizing for faster detection
        frame = cv2.resize(frame, (650,500))
        # using a greyscale picture for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
        print('Human Detected : ', len(boxes))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
