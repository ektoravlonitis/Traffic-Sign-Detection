# This is a Traffic Sign Information Project.

First run the ClassificationCNN.py which saves the model that will be used for the object detection methods.

### Overview
The Traffic Sign Information project aims to develop a deep neural network model for traffic sign classification. The primary goal is to create a system capable of accurately identifying and classifying various types of traffic signs in images. The project utilizes a dataset obtained from Kaggle, comprising over 50,000 images of traffic signs categorized into 43 classes.

### Project Steps
The project is divided into four main steps:

#### Data Exploration
The first step involves exploring the dataset, understanding its structure, and gaining insights into the distribution of images across different classes.
#### Model Building
The second step focuses on building a Convolutional Neural Network (CNN) model. The model architecture is designed to learn and extract relevant features from traffic sign images to achieve accurate classification.
#### Model Training
Once the model is built, it is trained and validated using the dataset to ensure its effectiveness in classifying traffic signs accurately.
#### Model Testing
In the fourth step, the trained model is tested using a separate test dataset. This step assesses the model's performance on unseen traffic sign images, providing insights into its real-world applicability.

### Techniques Employed
For traffic sign classification, multiple CNN models were implemented due to various setbacks and to achieve higher accuracy. For example, to improve accuracy in object detection, a model was trained only on half of the traffic sign classes using a brute force approach.

For object detection, three approaches were implemented:
#### Brute Force
This approach utilizes a sliding window technique to systematically scan the image. However, it is computationally expensive and can produce false positives.
#### Selective Search
This approach generates potential object regions based on similarity measures, reducing computational burden compared to the brute force approach.
#### YOLOv8
YOLOv8 is a state-of-the-art algorithm that divides the image into a grid and predicts objects in each grid cell. It has shown superior performance compared to other methods.
By employing these techniques, the Traffic Sign Information project aims to develop an accurate and efficient system for traffic sign classification and real-time detection.
