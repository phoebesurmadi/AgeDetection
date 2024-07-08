# Age Detection using OpenCV and Deep Learning

This project utilizes OpenCV and deep learning techniques to classify the age of individuals from facial images.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Data Augmentation](#data-augmentation)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
6. [Usage](#usage)


## Overview

Age detection is achieved through a pre-trained deep neural network model that analyzes facial features to predict the age of a person.

## Features

- **Age Classification:** Predict the age of individuals from images.
- **Python Implementation:** Implemented in Python for ease of use and integration.
- **Deep Learning:** Utilizes deep neural networks for accurate age prediction.

## Dataset

This project uses the UTKFace dataset (Inthewild/part 1) for training and testing the age detection model.

### About UTKFace Dataset
- **Source**: UTKFace dataset from the University of Tennessee, Knoxville
- **Contents**: Large-scale face dataset with long age span (range from 0 to 116 years old)
- **Labels**: Each image is labeled with age, gender, and ethnicity
- **Format**: Aligned and cropped faces
- **File Naming**: [age]_[gender]_[race]_[date&time].jpg

### Dataset Location
The dataset is located in the `data/UTKFace_subset` directory of this project.

### Usage
The `data_loader.py` script is configured to load and preprocess images from this dataset.

Note: Due to the large size of the dataset, it is not included in this repository. Please download it separately and place it in the appropriate directory.

## Data Augmentation

This project employs data augmentation techniques to enhance the model's performance and generalization. The following augmentations are applied during training:

1. Random horizontal flipping
2. Random brightness adjustment
3. Random contrast adjustment

### Rationale for Data Augmentation

Data augmentation is crucial for our age detection model for several reasons:

1. **Increased Dataset Variety**: These transformations effectively increase the size and diversity of our training dataset without requiring additional data collection.

2. **Improved Generalization**: By exposing the model to various image conditions, we enhance its ability to recognize faces and estimate ages accurately across different scenarios.

3. **Real-world Simulation**:
   - Horizontal flipping helps the model understand that face orientation doesn't affect age.
   - Brightness adjustments simulate different lighting conditions.
   - Contrast changes account for variations in image quality or camera settings.

4. **Overfitting Prevention**: Random transformations reduce the likelihood of the model memorizing specific images, encouraging it to learn more general age-related features.

5. **Bias Mitigation**: Augmentation can help address potential biases in the original dataset, such as consistent lighting conditions.

These augmentation techniques are implemented in the `augment_image` function within `data_loader.py` and are applied during the training process.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have the following installed:

- Python 3.12.4
- OpenCV (cv2)
- NumPy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/phoebesurmadi/AgeDetection.git
   cd AgeDetection
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

