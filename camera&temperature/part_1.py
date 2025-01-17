import cv2;

# add an IP adress that is shown in IP Webcam. For current desktop use value '0'
capture = cv2.VideoCapture('http://192.168.100.11:8080/video');

while True:
    ret, frame = capture.read(); #read frames in a loop
    if frame is not None:
        resized = cv2.resize(frame, (800, 550)); #resize the frame (desktop)
        cv2.imshow("Frame", resized); #display the frame on desktop

    key = cv2.waitKey(1); #to exit the frame press 'q'
    if key == ord("q"):
        break;

capture.release(); #release video capture object
cv2.destroyAllWindows(); # destroy all frame windows