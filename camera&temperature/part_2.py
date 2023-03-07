import cv2;
import numpy;

# add an IP adress
capture = cv2.VideoCapture('http://192.168.100.11:8080/video');

while True:
    ret, frame = capture.read();
    frame_copy = frame.copy();
    cv2.imshow("Show temperature", frame_copy);
    temp = 5; 
    cv2.putText(frame_copy, f'The temperature in the room is {temp}', (40,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (255,0,255), 2, cv2.LINE_AA);

    key = cv2.waitKey(1); 
    if key == ord("q"):
        break;

capture.release();
cv2.destroyAllWindows();