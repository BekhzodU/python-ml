import cv2
import pytesseract;

capture = cv2.VideoCapture(0);
#get the default size of the frame
WIDTH = capture.get(cv2.CAP_PROP_FRAME_WIDTH);
HEIGHT = capture.get(cv2.CAP_PROP_FRAME_HEIGHT);

while True:
    _, frame = capture.read();
    frame_copy = frame.copy();
    #configure size to reduce the size of the screen for digit prediction
    bbox_size = (250, 250);
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))];

    #crop and resize the copy of frame for prediction
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]];
    #change the color to grayscale for pytesseract and resize
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY);
    img_gray = cv2.resize(img_gray, (400, 400));

    #configuration for pytesseract to only work with digits
    conf = r'--oem 3 --psm 6 outputbase digits'
    #returns digits and their coordinates in the frame
    boxes = pytesseract.image_to_boxes(img_gray,config=conf);
    str = '';
    for b in boxes.splitlines():
        #take the value of digit only without coordinates
        b = b.split(' ');
        str += b[0];

    #shape green rectangle in the frame for digit recognition
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0,255,0),3);
    #write string in the frame
    cv2.putText(frame_copy, f'The temperature in the room is {str}', (90,90), cv2.FONT_ITALIC, 1.5, (0,0,255), 5, cv2.LINE_AA)
    cv2.imshow("Result", frame_copy);

    key = cv2.waitKey(1); 
    if key == ord("q"):
        break;

capture.release();
cv2.destroyAllWindows();