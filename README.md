# Food Image Classification System

    Note: This was a group coursework project created by me and two of my colleagues from university

---

![Application](https://github.com/user-attachments/assets/5d6c260d-43d2-41ca-938a-0f58962a4a03)

---

## Overview

A computer vision system for food recognition using different machine learning approaches, including Convolutional Neural Networks (CNN), Histogram of Oriented Gradients (HOG), and Capsule Networks.

## Project Structure

### CNN Implementation

  - CNN_training.py - Trains a CNN model on the Food-101-tiny dataset

  - CNN_loading.py - Loads the trained model and makes predictions on new images

  - CNN.py - Template CNN implementation with placeholder paths

### Alternative Approaches (my colleagues' part of the coursework)

  - Use_Histogram_of_Oriented_Gradients.py - HOG feature extraction with SVM classifier

  - Use_Capsule_Networks_CapsNets.py - Capsule Network implementation

## Dataset

The system is designed for the Food-101-tiny dataset containing 10 food categories:

  - apple_pie, bibimbap, cannoli, edamame, falafel

  - french_toast, ice_cream, ramen, sushi, tiramisu

## How to Use

### Training the CNN Model

  1. Place dataset in the appropriate directory structure

  2. Update file paths in CNN_training.py

  3. Run the training script:

  python CNN_training.py

### Making Predictions

  Use the trained model with CNN_loading.py:

  python CNN_loading.py

### HOG + SVM Approach

  Run the HOG implementation:

  python Use_Histogram_of_Oriented_Gradients.py

## Model Architecture

  - Convolutional Neural Network with 3 convolutional layers

  - Max pooling and dense layers for classification

  - Data augmentation (rotation, zoom, flipping)

  - Softmax output for multi-class classification

## Requirements

  - TensorFlow 2.x

  - scikit-learn

  - scikit-image

  - NumPy

  - Matplotlib

  - Pillow (PIL)
