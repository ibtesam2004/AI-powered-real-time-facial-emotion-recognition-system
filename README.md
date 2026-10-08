# AI-Powered Real-Time Facial Emotion Recognition System

An AI-powered facial emotion recognition system that detects human faces and classifies their facial expressions into seven emotion categories using a **Convolutional Neural Network (CNN)** and **OpenCV**.

The project supports both **real-time webcam emotion detection** and **static image-based emotion prediction**, with a pre-trained Keras model included for inference.

## 🚀 Features

- 🎭 Recognizes **7 facial emotions**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- 👤 Detects faces using **OpenCV Haar Cascade Classifier**
- 🧠 Classifies facial expressions using a **CNN built with Keras/TensorFlow**
- 📹 Supports **real-time webcam emotion recognition**
- 🖼️ Supports **emotion detection from static images**
- 📊 Uses image augmentation during model training
- ⚡ Performs predictions on selected video frames to improve real-time performance
- 📦 Includes a pre-trained `.h5` model for immediate testing

## 🧠 How It Works

The system follows a simple computer vision and deep learning pipeline:

```text
Camera / Image
      ↓
Face Detection
      ↓
Grayscale Conversion
      ↓
Face Cropping
      ↓
Resize to 48 × 48
      ↓
Pixel Normalization
      ↓
CNN Model
      ↓
Emotion Classification
      ↓
Emotion + Confidence Score
```

For webcam input, faces are detected frame-by-frame and the trained CNN predicts the most likely facial emotion.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN model development and inference |
| OpenCV | Face detection, image processing and webcam handling |
| NumPy | Numerical and array operations |
| Haar Cascade | Face detection |

## 🏗️ Model Architecture

The CNN processes grayscale facial images resized to **48 × 48 pixels**.

The implemented architecture includes:

- Convolutional layer — 32 filters
- Convolutional layer — 64 filters
- Max Pooling
- Dropout
- Convolutional layer — 128 filters
- Max Pooling
- Dropout
- Convolutional layer — 256 filters
- Max Pooling
- Dropout
- Flatten layer
- Dense layer — 512 neurons
- Dropout
- Output layer — 7 neurons with Softmax activation

The model is trained using:

```text
Optimizer: Adam
Loss: Categorical Crossentropy
Output: 7-class Softmax
Input: 48 × 48 × 1
Training Epochs: 30
Batch Size: 32
```

## 📚 Dataset

The repository contains training and testing image directories organized by emotion class.

The included dataset contains:

| Emotion | Training Images | Test Images |
|---|---:|---:|
| Angry | 3,995 | 958 |
| Disgust | 436 | 111 |
| Fear | 4,097 | 1,024 |
| Happy | 7,215 | 1,774 |
| Neutral | 4,965 | 1,233 |
| Sad | 4,830 | 1,247 |
| Surprise | 3,171 | 831 |
| **Total** | **28,709** | **7,178** |

Training images are augmented using rotation, shear, zoom and horizontal flipping to improve model robustness.

## 📁 Project Structure

```text
AI-Powered-Real-Time-Facial-Emotion-Recognition-System/
│
├── data/
│   ├── models/
│   │   └── model_file_30epochs.h5
│   │
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   │
│   ├── test/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   │
│   ├── haarcascade_frontalface_default.xml
│   ├── image1.jpg
│   ├── image2.png
│   ├── main.py
│   ├── test.py
│   └── testdata.py
│
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Powered-Real-Time-Facial-Emotion-Recognition-System.git
cd AI-Powered-Real-Time-Facial-Emotion-Recognition-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow keras opencv-python numpy
```

## ▶️ Usage

### Real-Time Webcam Detection

Run:

```bash
python data/test.py
```

The application will:

1. Open the system camera.
2. Detect faces in the video stream.
3. Process each detected face.
4. Predict one of the seven emotion classes.
5. Display the detected emotion and confidence score on the video feed.

Press:

```text
q
```

to exit the application.

### 🖼️ Static Image Detection

To test the system using the included image:

```bash
python data/testdata.py
```

The script loads `data/image2.png`, detects faces and displays the predicted emotion.

### 🧪 Train the Model

To train the CNN from the included dataset:

```bash
python data/main.py
```

The training script applies image augmentation and trains the model for **30 epochs**.

The generated model is saved as:

```text
model_file.h5
```

## 🔍 Emotion Labels

The classifier maps the model output to the following labels:

```python
{
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}
```

## 📈 Real-Time Performance Optimization

For webcam-based inference, the implementation performs model prediction every **5th frame** rather than predicting on every frame.

This reduces unnecessary computation while maintaining a responsive real-time experience.

## 🎯 Use Cases

This project demonstrates the practical application of computer vision and deep learning in areas such as:

- Human-computer interaction
- Emotion-aware applications
- AI-based behavioral analysis
- Educational demonstrations
- Computer vision research
- Interactive AI systems
- Real-time facial expression analysis


## 🔮 Future Improvements

Potential improvements include:

- Using a more advanced face detector such as **MTCNN, RetinaFace or MediaPipe**
- Evaluating modern CNN architectures
- Applying transfer learning
- Improving performance on minority classes such as **Disgust**
- Adding prediction smoothing across consecutive frames
- Developing a graphical user interface
- Adding emotion history and analytics
- Deploying the model as a web or desktop application
- Evaluating the model using precision, recall, F1-score and confusion matrices

## 🔐 Privacy & Responsible Use

This project is intended for **educational, research and demonstration purposes**.

Facial emotion predictions should be handled responsibly, particularly in applications involving sensitive personal information. Users should provide appropriate consent before processing facial images or webcam data.

## 👨‍💻 Author

**Ibtesam**

Computer Science & Engineering  
Interested in **Artificial Intelligence, Machine Learning, Full-Stack Development and Cybersecurity**

## ⭐ Support

If you find this project useful, consider giving the repository a **star ⭐** and sharing your feedback.

---

### 📌 Project Summary

> **An AI-powered real-time facial emotion recognition system built with Python, TensorFlow/Keras and OpenCV, combining Haar Cascade face detection with a CNN-based seven-class emotion classifier.**
