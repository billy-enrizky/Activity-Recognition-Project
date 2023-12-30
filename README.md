# Human Activity Recognition using CNN + LSTM Deep Learning Project

This project implements a Human Activity Recognition (HAR) system using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The goal is to recognize and classify human activities based on video input.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Action Recognition on YouTube Videos](#action-recognition-on-youtube-videos)
- [Streamlit User-Friendly Version](#streamlit-user-friendly-version)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Human Activity Recognition (HAR) is a field of study that involves the identification and classification of human activities based on data such as video sequences. In this project, we use a combination of CNN and LSTM layers to build an effective HAR model.

## Project Overview

The project is divided into two main parts:

1. **Part 1: CNN + LSTM Model Training**
   - Train a model using CNN and LSTM layers for Human Activity Recognition.
   - Plot model loss and accuracy curves.
   - Implement the Long-term Recurrent Convolutional Network (LRCN) approach.

2. **Part 2: Action Recognition on Videos**
   - Download YouTube videos for testing.
   - Create functions for action recognition in videos.
   - Evaluate the trained model on test videos.
   - Implement a user-friendly Streamlit version.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- MoviePy
- Pafy
- Matplotlib
- Streamlit (for Streamlit version)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Part 1: Model Training**

   - Open and run the Jupyter notebook for Part 1.
   - Follow the instructions in the notebook to train the CNN + LSTM model.

2. **Part 2: Action Recognition on Videos**

   - Open and run the Jupyter notebook for Part 2.
   - Follow the instructions in the notebook to download YouTube videos and perform action recognition.

## Models

- Two models are implemented in this project:
  1. ConvLSTM Model
  2. Long-term Recurrent Convolutional Network (LRCN) Model

- The models are trained for Human Activity Recognition.

## Evaluation

- Model performance is evaluated on test datasets.
- Loss and accuracy curves are plotted for model evaluation.

## Action Recognition on YouTube Videos

- Functions are provided to download YouTube videos for testing.
- Action recognition is performed on these videos using the trained model.

## Streamlit User-Friendly Version

- A Streamlit version is created for easy interaction.
- Users can recognize actions in a video with a single click.

## Contributing

Contributions are welcome! If you have suggestions or find issues, please open an [issue](<link-to-issues>) or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

```

Please make sure to replace `<repository-url>`, `<repository-directory>`, and `<link-to-issues>` with the appropriate values. Additionally, include the license file (e.g., `LICENSE`) in your project directory. Adjust the structure and content as needed for your specific project.
