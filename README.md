# Helmet-Detection-yolov5

This project focuses on training a custom YOLOv5 model to detect helmets and heads in images and videos. Using annotations for helmet detection, the model is trained on a dataset of images and then applied to detect helmets in new images or videos.

## Overview

This repository leverages the YOLOv5 architecture to train a helmet detection model. The process includes:
1. **Dataset Preparation**: Images and annotations are organized into training and validation sets.
2. **Training**: YOLOv5 is trained on the dataset with custom annotations for helmet detection.
3. **Detection**: The trained model is used to detect helmets in images and videos.
4. **Video Detection**: You can also apply the trained model to detect helmets in videos.

## Features

- Custom helmet detection model using YOLOv5.
- Preprocessing of image and label data for YOLOv5 compatibility.
- Training with custom labels for "Helmet" and "Head".
- Model inference on both images and videos.
- Easy integration with the YOLOv5 ecosystem.

## Technologies Used

- **Python 3.x**
- **YOLOv5** (Ultralytics)
- **PyTorch** for deep learning model training
- **OpenCV** for image and video processing
- **WandB** for experiment tracking (disabled in the notebook)
- **NumPy, Pandas, Matplotlib** for data manipulation and visualization

## Setup

### 1. Clone the Repository
To get started with the project, clone the repository:
met-detection-yolov5


### 2. Install Dependencies
Ensure you have Python 3.x installed 

This will install the following libraries:
- YOLOv5 dependencies (including `torch`, `opencv-python`, etc.)
- WandB (disabled in the notebook)
- Other necessary packages for model training and inference

### 3. Dataset Preparation
The dataset for training consists of images and labels in YOLO format.

The dataset should include both image files (e.g., `.jpg` or `.png`) and corresponding label files (e.g., `.txt`) where each line in a label file represents a bounding box and class label.

You can use the provided notebook to prepare your dataset and split it into training and validation sets. The dataset will be saved in the `dataset` directory.

### 4. Training the YOLOv5 Model
Once your dataset is ready, you can train the model using the following command:
```bash
!python train.py --img 415 --batch 10 --epochs 30 --data /path/to/my_data.yaml --weights yolov5s.pt --cache --workers 2
```
- `--img 415`: Image size for training.
- `--batch 10`: Batch size.
- `--epochs 30`: Number of epochs to train the model.
- `--data /path/to/my_data.yaml`: Path to your dataset YAML configuration file.
- `--weights yolov5s.pt`: Pre-trained YOLOv5 weights to start training from.
- `--cache --workers 2`: Caching and parallel data loading options.

### 5. Detection Inference on Images
After training, you can use the trained model to perform helmet detection on images. For example:
```bash
!python detect.py --source /path/to/image.jpg --weights runs/train/exp/weights/best.pt
```
This command will take an image (e.g., `/path/to/image.jpg`), apply the trained model, and save the detected image with bounding boxes.

### 6. Detection Inference on Videos
You can also use the trained model to detect helmets in videos. Just provide the video file path:
```bash
!python detect.py --source /path/to/video.mp4 --weights runs/train/exp/weights/best.pt
```
The model will process the video frame by frame, detecting helmets in each frame, and output the results.


## Files in the Repository

- **Annotation_helmet.ipynb**: Script to train the YOLOv5 model on custom dataset.
- **Video of output**
- **README.md**: This file with instructions and details on how to run the project.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. If you find any bugs or have feature requests, please open an issue.
