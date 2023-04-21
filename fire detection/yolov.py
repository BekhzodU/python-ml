import torch
import cv2

class FireDetection:
    
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        return model

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
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.25:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                value_str = str(row[4])
                start_index = value_str.index("(") + 1
                end_index = value_str.index(")")
                numeric_value = value_str[start_index:end_index]
                cv2.putText(frame, self.class_to_label(labels[i]) + " "+ numeric_value, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        while cap.isOpened():
            
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow("Fire Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Create a new object and execute.
fireDetection = FireDetection('fire.pt')
fireDetection()