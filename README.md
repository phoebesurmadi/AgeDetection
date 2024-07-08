# Age Detection using OpenCV and Deep Learning

This project utilizes OpenCV and deep learning techniques to classify the age of individuals from facial images.

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
The dataset is located in the `data/UTKFace` directory of this project.

### Usage
The `data_loader.py` script is configured to load and preprocess images from this dataset.

Note: Due to the large size of the dataset, it is not included in this repository. Please download it separately and place it in the appropriate directory.

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

