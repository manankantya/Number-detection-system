# ✨ Real-Time Digit Recognition with MNIST ✨

This project is a real-time digit recognition system using a Convolutional Neural Network (CNN) trained on the **MNIST dataset**.  
It allows you to draw/write digits in front of your webcam, and the model predicts them live with confidence scores.

---

## 🚀 Features
- **Enhanced CNN architecture** with multiple convolutional layers and dropout.
- **Data augmentation** for improved generalization.
- **Early stopping** to prevent overfitting.
- **Real-time digit recognition** using your webcam.
- **Prediction smoothing** to reduce flickering in outputs.

---

## 📂 Project Structure
├── Number_Detection.py    # Main script (model + real-time recognition)
├── enhanced_mnist_model.h5      # Saved trained model (auto-created after training)
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/real-time-digit-recognition.git
   cd real-time-digit-recognition

2.	Create a virtual environment (recommended):
python3 -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows

3. Install dependencies:
pip install -r requirements.txt

## Future Work
	•	Extend to EMNIST for character recognition.
	•	Add drawing pad interface for more controlled input.
	•	Implement adaptive thresholding for better performance under varied lighting.
