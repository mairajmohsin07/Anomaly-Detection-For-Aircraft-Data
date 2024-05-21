# Anomaly Detection for Aircraft Data

## Overview
This project focuses on detecting anomalies in turbine engine sensor data using two different machine-learning approaches: an Autoencoder and an Isolation Forest. The project involves pre-processing sensor data, training anomaly detection models, and evaluating their performance.

## Features
- Data pre-processing and scaling.
- Training an Autoencoder model for anomaly detection.
- Training an Isolation Forest model for anomaly detection.
- Calculating reconstruction errors for the Autoencoder model.
- Identifying and visualizing anomalies.
- Evaluation metrics for model performance.

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
   - [Install the required libraries](#install-the-required-libraries)
   - [Data Files](#data-files)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
   - [Loading and Pre-processing Data](#loading-and-pre-processing-data)
   - [Autoencoder Model](#autoencoder-model)
   - [Isolation Forest Model](#isolation-forest-model)
   - [Visualization and Evaluation](#visualization-and-evaluation)
5. [Results](#results)
6. [Contributions](#contributions)
7. [License](#license)

## Requirements
- Python 3.9
- pandas
- NumPy
- matplotlib
- Scikit-Learn
- Keras

## Setup

### Install the required libraries
You can install the necessary libraries using pip:
```sh
pip install pandas numpy matplotlib scikit-learn keras
```

### Data Files
Ensure you have the following data files in your working directory:
- `train_FD003.txt`
- `test_FD003.txt`

## Project Structure
```
├── anomaly_detection.ipynb    # Jupyter Notebook with the complete code
├── train_FD003.txt            # Training data file
└── test_FD003.txt             # Test data file
```

## Usage

### Loading and Pre-processing Data
1. Load the training data and drop irrelevant columns.
2. Scale the data using `StandardScaler`.

### Autoencoder Model
1. Define the architecture of the Autoencoder.
2. Compile and train the model on the scaled training data.
3. Load and scale the test data.
4. Calculate reconstruction errors and identify anomalies based on a threshold.
5. Visualize the reconstruction errors and anomalies.

### Isolation Forest Model
1. Train the Isolation Forest model on the scaled training data.
2. Predict the anomaly scores for the test data.
3. Calculate the threshold for anomaly detection.
4. Identify and visualize anomalies based on the scores.

### Visualization and Evaluation
1. Scatter plots of reconstruction errors.
2. Histogram of anomaly scores.
3. Calculation of evaluation metrics such as accuracy.

## Results
- The Autoencoder model identifies anomalies based on reconstruction errors.
- The Isolation Forest model detects anomalies based on anomaly scores.
- Visualization plots help in understanding the distribution of errors and scores.

## Contributions
Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.
