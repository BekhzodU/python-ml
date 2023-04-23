import telegram
from telegram.ext import Application, CommandHandler
import torch
import cv2
import time

def load_model(model_name):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
    return model

fire_model = load_model('fire.pt')

class FireDetection:
    
    def __init__(self, model):
        self.model = model
        self.classes = self.model.names

    def score_frame(self, frame):
        """
        scores the frame using yolo5 model.
        """
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        """
        fireDetected = False
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # confidence must be greater or equal than 50%
            if row[4] >= 0.5:
                fireDetected = True
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                value_str = str(row[4])
                start_index = value_str.index("(") + 1
                end_index = value_str.index(")")
                numeric_value = value_str[start_index:end_index]
                cv2.putText(frame, self.class_to_label(labels[i]) + " "+ numeric_value, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame, fireDetected

    async def return_fire_detected(self):
        result = False
        current_time = time.time()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        while time.time() - current_time < 15:
            _, frame = cap.read()
            if frame is not None:
                results = self.score_frame(frame)
                frame, fireDetected = self.plot_boxes(results, frame)
                cv2.imshow("Fire Detection", frame)
                if fireDetected:
                    result = True
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

        cap.release()
        cv2.destroyAllWindows()
        return result

  
# Function for detecting fire
async def is_fire_detected():
    fireDetected = await FireDetection(fire_model).return_fire_detected()
    return fireDetected

# Handler function for the /start command
async def start(update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi there! I will notify you in case of fire, I am developed by Ubaydullaev Bekhzod")

# Handler function for the /check command
async def check(update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait. Fire detection is in progress...")
    if await is_fire_detected():
       await context.bot.send_message(chat_id=update.effective_chat.id, text="Fire detected in the room!")
    else:
       await context.bot.send_message(chat_id=update.effective_chat.id, text="No fire detected in the room.")
    
  
# Create the Telegram bot and set up the command handlers
application = Application.builder().token("5953069246:AAF0tfVXsmN1lrZkVimqlWISn15m02ZQFfQ").build()
application.add_handler(CommandHandler('start', start))
application.add_handler(CommandHandler('check', check))

try:
# Start the bot using long polling
    application.run_polling(timeout=30)
finally:
    application.stop()