import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model

capture = cv2.VideoCapture(0);
model = load_model('digits.h5');

class obj:
    def __init__(self, num, y1, y2, x1, x2) -> None:
        self.num = num
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

def prediction(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    predict = model.predict(img)
    print(predict)
    prob = np.amax(predict)
    class_index = model.predict_classes(img)
    #class_index = np.argmax(predict, axis=1)
    result = class_index[0]
    if prob < 0.75:
        result = 0
        prob = 0
    return result, prob

objList = list()

while True:
    _, frame = capture.read();
    if frame is not None:
        frame_copy = frame.copy();
        frame_copy2 = frame.copy();
        img = frame_copy
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        hImg, wImg,_ = img.shape
        conf = r'--oem 3 --psm 6 outputbase digits'
        boxes = pytesseract.image_to_boxes(img,config=conf)


        for b in boxes.splitlines():
            b = b.split(' ')
            if(b[0].isdigit()):
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                img_cropped = img[hImg-h-10:hImg-y+10, x-10:w+10]
                objList.append(obj(b[0], hImg-h-10,hImg-y+10, x-10,w+10))

        res = ''
        for el in objList:
            result, prob = prediction(frame_copy2[el.y1:el.y2, el.x1:el.x2], model)
            if(prob != 0):
                res += str(result);
        
        cv2.putText(frame, f'The temperature in the room is {res}', (90,90), cv2.FONT_ITALIC, 1.5, (0,0,255), 5, cv2.LINE_AA)
        cv2.imshow("Result", frame);

        key = cv2.waitKey(1); 
        if key == ord("q"):
            break;

capture.release();
cv2.destroyAllWindows();

