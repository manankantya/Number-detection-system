import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# 1. Enhanced MNIST Model
# ======================
def create_enhanced_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                 loss='sparse_categorical_crossentropy',  
                 metrics=['accuracy'])
    return model

# Load or train model
try:
    model = load_model('enhanced_mnist_model.h5')
    print("Loaded pre-trained MNIST model")
except:
    print("Training enhanced MNIST model...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Enhanced preprocessing
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    # Enhanced data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1)
    datagen.fit(x_train)
    
    model = create_enhanced_model()
    
    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    model.fit(datagen.flow(x_train, y_train, batch_size=128),
              epochs=15,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])
    model.save('enhanced_mnist_model.h5')

# ======================
# 2. Real-time Prediction Setup
# ======================
cap = cv2.VideoCapture(0)
last_prediction = None
prediction_confidence = 0
prediction_history = []
smoothing_factor = 0.7  # For prediction smoothing

# ======================
# 3. Main Loop with Real-time Prediction
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Extract ROI (center of frame)
    h, w = frame.shape[:2]
    roi = gray[h//2-100:h//2+100, w//2-100:w//2+100]
    
    # Preprocess for prediction
    processed_roi = cv2.resize(roi, (28, 28))
    _, processed_roi = cv2.threshold(processed_roi, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Show processed digit
    processed_display = cv2.resize(processed_roi, (200, 200), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Processed Digit', processed_display)
    
    # Prepare for prediction
    prediction_input = processed_roi.astype('float32') / 255
    prediction_input = prediction_input.reshape(1, 28, 28, 1)
    
    # Predict
    predictions = model.predict(prediction_input, verbose=0)
    current_prediction = np.argmax(predictions)
    current_confidence = np.max(predictions)
    
    # Smooth predictions
    if len(prediction_history) > 0:
        last_prediction = prediction_history[-1][0]
        last_confidence = prediction_history[-1][1]
        smoothed_prediction = int(smoothing_factor * current_prediction + 
                                (1-smoothing_factor) * last_prediction)
        smoothed_confidence = smoothing_factor * current_confidence + \
                            (1-smoothing_factor) * last_confidence
    else:
        smoothed_prediction = current_prediction
        smoothed_confidence = current_confidence
    
    prediction_history.append((smoothed_prediction, smoothed_confidence))
    if len(prediction_history) > 5:  # Keep last 5 predictions
        prediction_history.pop(0)
    
    # Display
    cv2.rectangle(frame, (w//2-100, h//2-100), (w//2+100, h//2+100), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {smoothed_prediction}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {smoothed_confidence:.1%}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, 
               (0, 255, 0) if smoothed_confidence > 0.7 else (0, 0, 255), 2)
    cv2.putText(frame, "Press Q to quit", (10, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Real-time Digit Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()