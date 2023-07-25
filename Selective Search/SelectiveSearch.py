import cv2
import numpy as np
from skimage import img_as_float
from skimage.segmentation import felzenszwalb
import tensorflow as tf

# Define the class dictionary
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

def selective_search(image_path, scale=100, sigma=0.5, min_size=20):
    # Load image
    image = cv2.imread(image_path)
    # Convert the image to range 0 - 1
    image1 = img_as_float(image)
    # Perform Felzenszwalb segmentation
    segments = felzenszwalb(image1, scale, sigma, min_size)

    # Initialize a list to hold region proposals
    region_proposals = []

    # Loop over unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):
        # Construct a mask for the segment
        mask = np.zeros(image1.shape[:2], dtype="uint8")
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

    return image, image1, region_proposals


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

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
# Load the trained CNN model
#model = tf.keras.models.load_model('new_fresh_model_signs.h5')
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

num_classes = 43
# Load your trained CNN model
model = CNNModel(num_classes)  # Create an instance of your CNN model
model.load_state_dict(torch.load('model_all.pth'))  # Load the trained model state dict
model.eval()

# Call the Selective Search function
image, image1, proposals = selective_search('stop.jpg')
#image, image1, proposals = selective_search('stop3.jpg')

print(image1)
print(proposals)
# Define the threshold for the minimum proposal area
area_threshold = 500

# Create a copy of the original image to draw proposal boxes
image_proposals = image1.copy()

# Draw proposals larger than the area threshold
for (x1, y1, x2, y2) in proposals:
    area = (x2 - x1) * (y2 - y1)
    if area > area_threshold:
        cv2.rectangle(image_proposals, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Resize the image_proposals for smaller display
scale_percent = 50  # Adjust this value to change the size of the displayed image
width = int(image_proposals.shape[1] * scale_percent / 100)
height = int(image_proposals.shape[0] * scale_percent / 100)
image_proposals = cv2.resize(image_proposals, (width, height))

# Display the image with proposal boxes
cv2.imshow('Image with Proposal Boxes', image_proposals)
cv2.waitKey(0)
cv2.destroyAllWindows()



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
        output = model(image1)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = predicted.item()
    
    confidence_threshold = 0.8  # Adjust the threshold as needed

    if confidence.item() > confidence_threshold:
        print(f"Predicted label: {predicted_label}")
        return predicted_label
    else:
        print("No traffic sign detected.")
        return None

input_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract the proposal regions from the original image and resize
detected_signs = []

proposals_regions = []
proposals_to_evaluate = []
for (x1, y1, x2, y2) in proposals:
    area = (x2 - x1) * (y2 - y1)
    if area > area_threshold:
        roi = input_image_rgb[y1:y2, x1:x2]
        predicted_class = classify_traffic_sign(roi)

        # If the predicted class indicates a traffic sign, add it to the list of detected signs
        if predicted_class is not None:
            detected_signs.append((x1, y1, x2, y2, predicted_class))



for box in detected_signs:
    x, y, w, h, predicted_class = box
    #x, y, roi, _ , predicted_class = box
    # Extract the width and height of the ROI
    #h, w, _ = roi.shape

    # Convert the indexing values to integers
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    # Extract the region of interest (ROI) from the original image
    #roi = input_image_rgb[y:y+h, x:x+w]
    
    label = classes[predicted_class]
    # Print the predicted class label
    print("Detected Traffic Sign:", label)

    # Draw the bounding box and display the image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add the label text to the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x
    text_y = y - text_size[1] - 5
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

cv2.imshow('Image with Lines on Detected Traffic Signs', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
