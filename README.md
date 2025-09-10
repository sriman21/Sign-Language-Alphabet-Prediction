
# 🧠 Sign Language Alphabet Prediction

This project builds and trains a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of the American Sign Language (ASL) alphabet. It achieves high accuracy and includes visualisations to evaluate model performance.

## 📂 Dataset

- Source: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Contains 29 classes: A-Z, 'del', 'nothing', 'space'
- Images are resized to **64x64** for training

## 🧪 Preprocessing

- Normalised pixel values using `ImageDataGenerator`
- Split into training (80%) and validation (20%) sets
- Augmented data for better generalisation

# 📈 Performance
-Final training accuracy: ~99.7%

-Validation accuracy: ~72%

-Test accuracy: 100% on sample test set

# Visualisations include:

-Accuracy over epochs

-Confusion matrix

-Class-wise F1 scores

-Sample predictions

# 🔍 Evaluation
-Classification report using sklearn

-Confusion matrix heatmap with seaborn

-Visual comparison of predicted vs actual gestures

# 🖼️ Inference
-Predicts gestures from single images or batches

-Supports .jpg and .png formats

-Example usage:

python
predict_gesture("/path/to/image.jpg")

# 💾 Model Saving
-Saved as signlanguage_model.h5

-Recommended format: .keras for future compatibility

## 🏗️ Model Architecture

```python
Sequential([
  Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
  MaxPooling2D(2,2),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(29, activation='softmax')
])

