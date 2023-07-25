# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:43:12 2023

@author: he_98
"""

# Object detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

classes = {0: 'Speed limit (50km/h)',
 1: 'Speed limit (60km/h)',
 2: 'Speed limit (70km/h)',
 3: 'Speed limit (80km/h)',
 4: 'End of speed limit (80km/h)',
 5: 'Speed limit (100km/h)',
 6: 'Priority road',
 7: 'Yield',
 8: 'Stop',
 9: 'Dangerous curve left',
 10: 'Double curve',
 11: 'Beware of ice/snow',
 12: 'Wild animals crossing',
 13: 'Turn right ahead',
 14: 'Turn left ahead',
 15: 'Go straight or right',
 16: 'Keep right',
 17: 'Keep left',
 18: 'Roundabout mandatory'}

# Function to perform object detection using brute force
def detect_traffic_signs(image):
    # Parameters for sliding window
    #window_size = (150, 150)  # Size of the sliding window
    window_size = (100, 100)  # Size of the sliding window
    stride = 25  # Stride for window movement

    detected_signs = []

    # Get image dimensions
    height, width, _ = image.shape
    #print(height)
    #print(width)
    # ONLY THE TOP 50 % OF THE PICTURE
    # Slide the window across the image
    for y in range(0, int(0.5*height) - window_size[1], stride):
        #print(y)
        for x in range(0, width - window_size[0], stride):
            # Extract the current window from the image
            window = image[y:y+window_size[1], x:x+window_size[0]]
            #print(y)

            #print(window.shape)
            # Perform traffic sign classification on the window using your CNN model
            #predicted_class = 3
            predicted_class = classify_traffic_sign(window)

            # If the predicted class indicates a traffic sign, add it to the list of detected signs
            if predicted_class is not None:
                detected_signs.append((x, y, window_size[0], window_size[1], predicted_class))
    
    return detected_signs


import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image


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


num_classes = 19
# Load your trained CNN model
model = CNNModel(num_classes)  # Create an instance of your CNN model

model.load_state_dict(torch.load('modelBrute.pth'))  # Load the trained model state dict
model.eval()  # Set the model to evaluation mode

# Function to classify traffic signs using your pre-existing CNN model
def classify_traffic_sign(image):
    # Preprocess the image (resize, normalize, etc.) before passing it to the model
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = Image.fromarray(image)  # Convert NumPy array to PIL image
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        #print(probabilities)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = predicted.item()
    
    confidence_threshold = 0.9999999  # Adjust the threshold as needed

    if confidence.item() > confidence_threshold:
        print(f"Predicted label: {predicted_label}")
        return predicted_label
    else:
        print("No traffic sign detected.")
        return None


# Load the video
video_path = 'simulation_video.mp4'
cap = cv2.VideoCapture(video_path)
start_frame = 0

# Create a loop to read and process each frame of the video
frame_counter = 0
# Create a loop to read and process each frame of the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # Increment the frame counter
    frame_counter += 1    
    # Check if the frame was read successfully
    if not ret:
        break

    # Skip frames until the desired starting frame is reached
    if frame_counter < start_frame:
        continue
    # Rotate the frame by 180 degrees
    #rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Convert the rotated frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform object detection using brute force
    detected_boxes = detect_traffic_signs(frame_rgb)
    
    # Classify the detected traffic signs using your CNN model
    for box in detected_boxes:
        x, y, w, h, predicted_class = box

        # Extract the region of interest (ROI) from the original frame
        roi = frame_rgb[y:y+h, x:x+w]
        
        label = classes[predicted_class]

        # Print the predicted class label
        print("Detected Traffic Sign:", label)

        # Draw the bounding box on the frame
        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Define font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2  # Increase font size
        font_thickness = 2
        
        # Get text size
        #label = "Sample Text"
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Set text position
        text_x = x
        text_y = y - text_size[1] - 5
        
        # Draw text on the image
        cv2.putText(frame_rgb, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    # Display the frame with detected traffic signs
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()