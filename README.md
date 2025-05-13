# Kidney Stone Detection Using CNN

## Overview
This project develops a Convolutional Neural Network (CNN) to automatically detect kidney stones in CT scan images. By leveraging advanced image preprocessing techniques (Gaussian smoothing and Canny edge detection), the model achieves high accuracy in distinguishing between images with kidney stones and normal kidneys. The dataset, sourced from GitHub, contains approximately 1600 CT images. The model achieves a validation accuracy of 95%, with precision, recall, and F1-score metrics indicating robust performance.

## Project Features
- Preprocessing: Converts CT images to grayscale, applies Gaussian blur and Canny edge detection, and normalizes them for CNN input.
- CNN Architecture: Custom model with three Conv2D layers (32, 64, 128 filters), max-pooling, dropout, and dense layers with L2 regularization.
- Training: Uses Adam optimizer, binary cross-entropy loss, class weights for imbalanced data, and early stopping to prevent overfitting.
- Evaluation: Reports accuracy (95%), precision (93.75%), recall (88.24%), F1-score (90.91%), and confusion matrix.
- Visualization: Displays original images, edge-detected images, and predictions, along with accuracy/loss curves and performance metrics.

## Dataset
The dataset is sourced from the [KidneyStoneDetection GitHub repository](https://github.com/muhammedtalo/Kidney_stone_detection) and contains ~1600 CT scan images:
- Classes: Kidney stones (KS.png) and normal kidneys (N.png).
- Size: 608 MB.
- Directory: Images are expected in a folder named KidneyStones in the project root.

## Prerequisites
To run this project, you need:
- Python 3.9+
- Dependencies (listed in requirements.txt):

  numpy  matplotlib  opencv-python  scikit-learn  tensorflow  seaborn
- Hardware: A GPU is recommended for faster training, but a CPU is sufficient.
- Dataset: Download the dataset from the [GitHub repository](https://github.com/muhammedtalo/Kidney_stone_detection) and place it in a folder named KidneyStones.

## Installation
- Clone the Repository:
bash
git clone https://github.com/Priya-49/Kidney-Stone-Detection
cd Kidney-Stone-Detection


Create a Virtual Environment (optional but recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
bash
pip install -r requirements.txt


Prepare the Dataset:
Download the dataset from the source.
Extract it and ensure the images are in a folder named KidneyStones in the project root.



## Usage

### Run the Code:
Open the Jupyter notebook code.ipynb:
bash
jupyter notebook code.ipynb

Execute all cells to train the model, evaluate it, and test on a single image.
Alternatively, run the script directly:
bash
python main.py  # If you convert the notebook to a .py file


Test a Single Image:
Modify the maincode() call in the notebook to specify a test image path, 
e.g.
bash
:maincode("KidneyStones/43KS.png")

The output will display the original image, edge-detected image, and prediction ("yes" for stone, "no" for normal).


## Expected Output
Training logs with epoch-wise accuracy/loss.

Validation metrics: accuracy, precision, recall, F1-score.

Confusion matrix heatmap.

Bar chart of performance metrics.

Visualization of a single image prediction saved as output.png.



## Results
The model was trained for 45 epochs with a batch size of 115, achieving:

Validation Accuracy: 95.03%

Validation Loss: 0.3097

Confusion Matrix:
[[45  3]
[ 6 107]]

![image](https://github.com/user-attachments/assets/2af765b0-8e1f-4114-b041-8f86c4d155a7)

True Positives (TP): 107 (correctly detected stones)

True Negatives (TN): 45 (correctly detected normal)

False Positives (FP): 3 (incorrectly flagged as stones)

False Negatives (FN): 6 (missed stones)


### Metrics:

Precision: 93.75% (high reliability when predicting stones)
Recall: 88.24% (good at catching most stones)
F1-Score: 90.91% (balanced performance)
![image](https://github.com/user-attachments/assets/18ba2849-5473-408a-811c-d6899097d949)



### Visualizations:

![image](https://github.com/user-attachments/assets/974e0314-2914-4418-9fd4-c20e46e961c6)

![image](https://github.com/user-attachments/assets/749f447b-0526-4c85-8379-f95a8fcd582b)


### Project Structure

bash
Kidney-Stone-Detection/
│
├── KidneyStones/           # Folder containing CT scan images
├── code.ipynb              # Main Jupyter notebook with code
├── requirements.txt        # List of dependencies
├── output.png              # Sample output image (generated after running)
├── README.md               # This file



## References

Dataset: https://github.com/muhammedtalo/Kidney_stone_detection
Tamilselvi, T., "Computer Aided Diagnosis System for Stone Detection," 2011.
Mohan, M. B., et al., "Automated Detection of Kidney Stone Using Deep Learning Models," 2023.
TensorFlow Documentation: tensorflow.org

## Authors

P. Priyanka, 
P. Tejaswani,
P. Bhagya Rani,
M. Sai Praneeth

Institution: Vignan's Institute of Information Technology (Autonomous), Visakhapatnam, India

Guide: Mrs. Ch. Sridevi
