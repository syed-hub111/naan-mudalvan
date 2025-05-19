# 🏙️ Urban Intelligence from CSV Data: Land Use Classification & Traffic Forecasting

This project applies deep learning techniques to two critical urban planning tasks using **realistic CSV datasets**:

- 🏡 **Land Use Classification** from image data stored in CSV format
- 🚦 **Traffic Volume Prediction** using time-series data from CSV

---

## ✅ Features

- 📂 Loads and processes image and traffic datasets from **CSV files**
- 🧠 Classifies land use images into **Residential, Commercial, Industrial** using CNN
- 📈 Predicts traffic volume trends using **LSTM-based** sequence learning
- 📊 Visualization of model predictions vs actual values

---

## 🛠 Technology Used

- **Language:** Python 3
- **Libraries:**
  - `NumPy`, `Pandas` – Data manipulation
  - `Matplotlib` – Data visualization
  - `TensorFlow/Keras` – Deep learning models
  - `scikit-learn` – Dataset splitting and preprocessing

---

## ⚙️ How It Works

### 1. Land Use Classification
- Image pixel data is loaded from a CSV (`land_use_images.csv`), reshaped to `(64, 64, 3)`.
- Labels are one-hot encoded for CNN training.
- A **Convolutional Neural Network** is trained to classify images into one of three land use categories.
- Predictions are visualized with side-by-side comparisons of predicted vs actual classes.

### 2. Traffic Volume Prediction
- Time-series data is loaded from a CSV (`traffic_data.csv`) containing traffic volume.
- Sliding window approach is used to convert data into sequences of 10 time steps.
- An **LSTM** model is trained to forecast future traffic volume.
- Actual and predicted values are plotted to show model accuracy.

---

## 🧠 ML Techniques Used

- **CNN (Convolutional Neural Network):** For spatial pattern recognition in land use images.
- **LSTM (Long Short-Term Memory):** For capturing temporal patterns in traffic volume data.

---

## 🎯 Objective

The goal is to build a prototype system that demonstrates how deep learning models can:
- Automate land use classification from image data.
- Predict traffic volume trends based on historical patterns.

---

## 🖱️ Controls

- **Modify Input CSVs:**
  - `land_use_images.csv`: Should have 64x64x3 image pixel values with a `label` column.
  - `traffic_data.csv`: Should contain a single `volume` column of traffic values.

- **Adjust Parameters:**
  - Change `window` size for LSTM (currently 10)
  - Increase `epochs` for better model accuracy
  - Tune CNN architecture for deeper classification tasks

---

## 🏋️ Model Training

### CNN (for Image Classification)
- Input: `(64, 64, 3)` RGB images
- Architecture:
  - 2 Conv2D layers + MaxPooling
  - Flatten + Dense layers
  - Output: Softmax (3 classes)

### LSTM (for Traffic Forecasting)
- Input: `(10, 1)` time-step sequences
- Architecture:
  - LSTM(50 units) → Dense(1)
  - Output: Predicted traffic volume

---

## 📈 Output Explanation

### Land Use Classification
- Displays 6 test images:
  - **Predicted class**
  - **Actual class**

### Traffic Volume Prediction
- Line plot with:
  - Blue line: Actual volume
  - Orange line: Predicted volume
- Demonstrates model’s ability to follow traffic trends

---

## 📦 Data Collection

- **land_use_images.csv:**
  - Each row represents a flattened RGB image (shape 64x64x3) with a class label (0=Residential, 1=Commercial, 2=Industrial)

- **traffic_data.csv:**
  - A single-column CSV containing continuous traffic volume values

You may simulate or replace this data with real-world sources like:
- Satellite image exports
- Transportation department traffic APIs

---

## 📄 License

This project is released under the MIT License. Use it freely for education or prototyping.

---

## 🙌 Acknowledgments

Special thanks to the open-source community for providing robust tools for AI development. You can extend this project with:
- Satellite imagery (Sentinel, Google Earth Engine)
- Traffic sensors, IoT data, or public city datasets

---

