import cv2
import numpy as np
from skimage import img_as_float
from skimage.segmentation import felzenszwalb
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image

'''
Selective search
'''
def selective_search(image, scale=100, sigma=0.5, min_size=20):
    # Convert the image to range 0 - 1
    if image.dtype != np.float64:
        image = img_as_float(image)
    # Perform Felzenszwalb segmentation
    segments = felzenszwalb(image, scale, sigma, min_size)
    

    # Initialize a list to hold region proposals
    region_proposals = []

    # Loop over unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):
        # Construct a mask for the segment
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out non-rectangular regions and small regions
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 4:  # Consider only rectangular regions
                (x, y, w, h) = cv2.boundingRect(approx)
                if w * h > 100:  # Filter out small regions based on size
                    region_proposals.append([x, y, x + w, y + h])

    # Merge overlapping boxes
    region_proposals = merge_boxes(region_proposals)

    return image, region_proposals
def merge_boxes(boxes, iou_threshold=0.5):
    merged_boxes = []
    while boxes:
        main_box = boxes.pop(0)
        boxes = [box for box in boxes if IoU(main_box, box) < iou_threshold]
        merged_boxes.append(main_box)
    return merged_boxes


def IoU(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xi1, yi1, xi2, yi2 = max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)
    intersection = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union = box1_area + box2_area - intersection

    return intersection / union

'''
CNN MODEL
'''
classes = {0: 'Speed limit (50km/h)',
 1: 'Speed limit (60km/h)',
 2: 'Speed limit (70km/h)',
 3: 'Speed limit (80km/h)',
 4: 'End of speed limit (80km/h)',
 5: 'Speed limit (100km/h)',
 6: 'Stop',
 7: 'Dangerous curve left',
 8: 'Dangerous curve right',
 9: 'Double curve',
 10: 'Beware of ice/snow',
 11: 'Wild animals crossing',
 12: 'Turn right ahead',
 13: 'Turn left ahead',
 14: 'Ahead only',
 15: 'Go straight or right',
 16: 'Go straight or left',
 17: 'Keep right',
 18: 'Keep left',
 19: 'Roundabout mandatory'}

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
def classify_traffic_sign(roi):
    # Convert the model's parameters to float
    model_cnn = model_cnn.float()
    # Resize to the input size expected by your model and convert to PyTorch tensor
    roi = cv2.resize(roi, (64, 64))  # Change the size if necessary
    roi = transforms.ToTensor()(roi).float()
    roi = roi.unsqueeze(0)  # Add the batch dimension

    # If you're using a GPU, move the tensor to GPU memory
    if torch.cuda.is_available():
        roi = roi.cuda()
        model_cnn = model_cnn.cuda()
    else:
        model_cnn = model_cnn.cpu()

    # Use the model to make predictions
    predictions = model_cnn(roi.float())
    predicted_class = torch.argmax(predictions, dim=1)
    confidence = torch.nn.functional.softmax(predictions, dim=1)[0][predicted_class]

    # Convert the tensor to CPU memory (from GPU if necessary) and to numpy
    predicted_class = predicted_class.detach().cpu().numpy()[0]
    confidence = confidence.detach().cpu().numpy()

    return classes[predicted_class], confidence

num_classes = 19
# CNN
model_cnn = CNNModel(num_classes) # num_classes should be defined
model_cnn = model_cnn.float()  # Convert the model's parameters to float
model_cnn.load_state_dict(torch.load('model_20.pth')) # Load the trained model state dict
model_cnn.eval()

# Open the video file
video = cv2.VideoCapture(0)

# Get the video dimensions and frame rate
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the output video
output_file = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Apply the selective search function to the frame
    frame, proposals = selective_search(frame)

    # Define the threshold for the minimum proposal area
    area_threshold = 5000

    # Extract the proposal regions from the frame and resize
    proposals_regions = []
    proposals_to_evaluate = []
    for (x1, y1, x2, y2) in proposals:
        area = (x2 - x1) * (y2 - y1)
        if area > area_threshold:
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (64, 64)) # Change the size to 64x64
            roi = transforms.ToTensor()(roi) # Convert to PyTorch tensor
            roi = roi.unsqueeze(0) # Add batch dimension
            proposals_regions.append(roi)
            proposals_to_evaluate.append((x1, y1, x2, y2))

    # Set a confidence threshold for detection
    confidence_threshold = 0.9

    # Draw bounding boxes on the detected traffic signs
    for idx, roi in enumerate(proposals_regions):
        roi = roi.float()  # Convert the input tensor to float
        predictions = model_cnn(roi)
        predicted_class = torch.argmax(predictions, dim=1)
        confidence = torch.nn.functional.softmax(predictions, dim=1)[0][predicted_class]
        
        if confidence > confidence_threshold:
            (x1, y1, x2, y2) = proposals_to_evaluate[idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            traffic_sign_label = classes[predicted_class.detach().cpu().numpy()[0]]
            cv2.putText(frame, traffic_sign_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Write the frame to the output video
    output_video.write(frame)

    # Show the frame
    cv2.imshow('Video', frame)
    
    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and the output video
video.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
