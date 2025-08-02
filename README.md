
# Traffic Management System using YOLOv9 and Computer Vision

This project implements a **Traffic Management System** using machine learning and computer vision to detect vehicles and manage traffic signals dynamically. The system utilizes the **YOLOv9** model for real-time vehicle detection and adjusts traffic signals based on the density of vehicles at each junction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [YOLOv9 Model](#yolov9-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to automate traffic signal control using a deep learning-based approach for detecting vehicles at intersections. The system identifies vehicle density using real-time video feeds and intelligently adjusts the traffic lights to improve the flow of traffic.

By integrating **YOLOv9** for vehicle detection, this system provides:
- Real-time detection of vehicles at traffic signals
- Dynamic traffic light adjustments based on vehicle density
- Improved traffic flow and reduced congestion

## Features

- **Vehicle Detection**: Detects cars, bikes, trucks, and other vehicles in real-time using the YOLOv9 model.
- **Traffic Signal Management**: Automatically adjusts signal timings based on traffic density at each intersection.
- **Efficient Resource Utilization**: Minimizes wait times for vehicles, reducing fuel consumption and emissions.
- **Scalable System**: Can be easily integrated into existing traffic control infrastructures.
- **Customizable Parameters**: You can modify the system to prioritize emergency vehicles or public transport.

## Installation

### Prerequisites
- Python 3.12
- YOLOv9 (pre-trained model)
- OpenCV
- NumPy
- PyTorch (for running the YOLOv9 model)
- Flask (for the web-based interface)
- Any video feed (camera or recorded footage)

### Steps to Install

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/traffic-management-yolov9.git
    cd traffic-management-yolov9
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv9 weights:
    Download the pre-trained YOLOv9 weights from the official source or link provided in this repository, and place them in the `models/` directory.

## Usage

1. **Running the Traffic Management System:**
    - To start the system, run the following command:
    ```bash
    python manage_traffic.py
    ```

2. **Input Feed:**
    - You can provide a real-time video feed from traffic cameras or use recorded footage for simulation.

3. **Web Interface:**
    - The system comes with a web-based interface where you can monitor the status of the signals in real-time.

## YOLOv9 Model

### What is YOLOv9?

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. YOLOv9 is the latest iteration and improves accuracy and speed, making it suitable for real-time traffic management applications. In this project, we leverage YOLOv9 to detect vehicles at intersections and dynamically control traffic signals.

### Training and Customization

The model can be retrained or fine-tuned for specific traffic conditions or regions if required. For custom training, follow the steps outlined in the `models/README.md`.

## Contributing

We welcome contributions! If you want to contribute to this project, please fork the repository and submit a pull request.

Steps to contribute:
1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Feature description"
    ```
4. Push to your branch:
    ```bash
    git push origin feature-name
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
