import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image

# classes and model definition
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicle with a weight greater than 3.5 tons'
}

# classes = {0: 'Speed limit (50km/h)',
#  1: 'Speed limit (60km/h)',
#  2: 'Speed limit (70km/h)',
#  3: 'Speed limit (80km/h)',
#  4: 'End of speed limit (80km/h)',
#  5: 'Speed limit (100km/h)',
#  6: 'Stop',
#  7: 'Dangerous curve left',
#  8: 'Dangerous curve right',
#  9: 'Double curve',
#  10: 'Beware of ice/snow',
#  11: 'Wild animals crossing',
#  12: 'Turn right ahead',
#  13: 'Turn left ahead',
#  14: 'Ahead only',
#  15: 'Go straight or right',
#  16: 'Go straight or left',
#  17: 'Keep right',
#  18: 'Keep left',
#  19: 'Roundabout mandatory'}


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Adjusted dimensions
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        
        self.dropout3 = nn.Dropout(0.5)  # Additional dropout layer

        
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)
        out = self.dropout1(out)

        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.maxpool2(out)
        out = self.dropout2(out)  # Apply the additional dropout layer
        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        
        return out

# Function to classify traffic signs using your pre-existing CNN model
def classify_traffic_sign(image1):
    # Preprocess the image (resize, normalize, etc.) before passing it to the model
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image1 = Image.fromarray(image1)  # Convert NumPy array to PIL image
    image1 = transform(image1)
    image1 = image1.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model_cnn(image1)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = predicted.item()
    
    confidence_threshold = 0.8  # Adjust the threshold as needed

    if confidence.item() > confidence_threshold:
        print(f"Predicted label: {predicted_label}")
        return predicted_label, confidence.item()
    else:
        print("No traffic sign detected.")
        return None, None
    # return predicted_label, confidence.item()
    

num_classes = 43

# YOLO
model_yolo = YOLO(r"yolo_model.pt")
cap = cv2.VideoCapture('ts.mp4')
# cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# CNN
model_cnn = CNNModel(num_classes) # num_classes should be defined
model_cnn.load_state_dict(torch.load('model_all.pth')) # Load the trained model state dict
model_cnn.eval()

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # video codec
fps = cap.get(cv2.CAP_PROP_FPS)  # get fps from input video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

frame_count = 0
frame_step = 1

while True:
    _, frame = cap.read()
    frame_count += 1

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_yolo = model_yolo.predict(img)
    annotator = Annotator(frame)

    if frame_count % frame_step == 0:  # Classify every 5 frames
        detected_labels = []
        confidences = []

        for r in results_yolo:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = [int(coord) for coord in b]
                
                # Expand the bounding box dimensions
                expansion_factor = 1.6
                w = x2 - x1
                h = y2 - y1
                x1 = max(0, int(x1 - (w * (expansion_factor - 1) / 2)))
                y1 = max(0, int(y1 - (h * (expansion_factor - 1) / 2)))
                x2 = min(frame.shape[1], int(x2 + (w * (expansion_factor - 1) / 2)))
                y2 = min(frame.shape[0], int(y2 + (h * (expansion_factor - 1) / 2)))

                detected_object = frame[y1:y2, x1:x2]
                traffic_sign_label, conf = classify_traffic_sign(detected_object)

                if traffic_sign_label is not None:
                    detected_labels.append(classes[traffic_sign_label])
                    confidences.append(conf)
                    annotator.box_label([x1, y1, x2, y2], classes[traffic_sign_label])
                else:
                    annotator.box_label([x1, y1, x2, y2])

        # Multiple predictions for the frame
        combined_text = ' + '.join(detected_labels)
        combined_conf = ' + '.join([f"{conf:.2f}" for conf in confidences])

        offset = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, combined_text, (20, offset), font, 1, (0, 255, 0), 3)
        cv2.putText(frame, combined_conf, (20, offset + 30), font, 1, (0, 0, 255), 3)

    frame = annotator.result()
    cv2.imshow('Traffic Sign Detection', frame)

    # Write the frame into the output file
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
out.release()
cv2.destroyAllWindows()