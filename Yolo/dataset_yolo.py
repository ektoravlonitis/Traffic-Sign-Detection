import os
import shutil
from sklearn.model_selection import train_test_split

# Directory containing all images and labels
source_dir = r'C:\Users\spika\Desktop\yolov5Try\ts\ts'

# Destination directories
train_dir = r'C:\Users\spika\Desktop\yolov8\train'
test_dir = r'C:\Users\spika\Desktop\yolov8\test'

# Create destination directories if they don't exist
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# Get all file names in source directory
files = os.listdir(source_dir)

# Separate images and labels
images = [file for file in files if file.endswith('.jpg')]
labels = [file for file in files if file.endswith('.txt')]

# Pair each image with its label
pairs = list(zip(sorted(images), sorted(labels)))

# Split data into training and testing
train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

# Function to move files
def move_files(pairs, destination):
    for img, lbl in pairs:
        shutil.move(os.path.join(source_dir, img), os.path.join(destination, 'images', img))
        shutil.move(os.path.join(source_dir, lbl), os.path.join(destination, 'labels', lbl))

# Move training files
move_files(train_pairs, train_dir)

# Move testing files
move_files(test_pairs, test_dir)
