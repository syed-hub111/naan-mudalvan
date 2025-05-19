# 🏙️ Urban Intelligence: Land Use Classification & Traffic Prediction

This project demonstrates how deep learning can be applied to urban planning tasks, specifically focusing on:

- Image-based **Land Use Classification**
- Time-series **Traffic Volume Prediction**

---

## ✅ Features

- 🏡 Simulated classification of **Residential**, **Commercial**, and **Industrial** land use via synthetic satellite-like images.
- 🚦 Forecasts future **Traffic Volume** using LSTM based on simulated data patterns.
- 📊 Visualization of classification results and traffic forecasts for intuitive understanding.
- 🔁 Fully reproducible and customizable data generation and model training pipelines.

---

## 🛠 Technology Used

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - `NumPy`, `Matplotlib` – Data processing and visualization
  - `TensorFlow/Keras` – Deep learning models (CNN, LSTM)
  - `scikit-learn` – Data splitting and preprocessing

---

## ⚙️ How It Works

### Land Use Classification:
1. Synthetic RGB images are generated for each land type.
2. A **Convolutional Neural Network (CNN)** is trained to classify images into 3 categories.
3. Model predictions are visualized with side-by-side actual labels.

### Traffic Volume Prediction:
1. Simulated traffic data is generated using sine waves and random noise.
2. Data is prepared into sequences (sliding window format).
3. An **LSTM model** predicts future traffic values based on historical patterns.
4. Actual vs. predicted values are plotted.

---

## 📦 Data Collection

- **Land Use Images:** Synthetic data generated with custom rules to simulate color patterns for each class.
- **Traffic Volume Data:** Generated programmatically using sine functions plus noise to simulate real-world trends.
- **No external datasets** are used – this makes the project fully self-contained and easy to run anywhere.

---

## 🎯 Objective

The objective is to prototype AI models that can assist in:
- Automatic categorization of land use types for satellite or drone imagery.
- Predictive modeling for traffic management and congestion planning.

---

## 🎮 Controls

- Modify `num_samples`, `img_size`, or traffic `time_steps` to simulate larger datasets.
- Adjust model architecture or training `epochs` for experimentation.
- Customize the color/textural rules in `generate_class_image()` for more detailed simulations.

---

## 🤖 ML Techniques Used

- **CNN (Convolutional Neural Network)** – For spatial feature extraction in land use image classification.
- **LSTM (Long Short-Term Memory)** – For sequence learning in traffic volume forecasting.

---

## 🏋️ Model Training

### CNN
- Input Shape: (64, 64, 3)
- Layers: `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`
- Output: 3-class softmax for Residential, Commercial, Industrial

### LSTM
- Input Shape: (10, 1)
- Layers: `LSTM(50)`, `Dense(1)`
- Output: Single scalar prediction (traffic volume at future time step)

Training for both models runs for **5 epochs**, but can be extended for better performance.

---

## 📈 Output Explanation

### Land Use Classification
- A set of 6 randomly selected test images are displayed.
- Each image shows:
  - **Predicted class**
  - **Actual class**

### Traffic Volume Prediction
- A line plot comparing:
  - **Actual traffic values**
  - **Predicted values** from LSTM

Both outputs demonstrate model performance visually and highlight the power of deep learning in urban planning applications.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

This project is a conceptual demo and can be extended to real-world urban datasets such as:
- Satellite imagery from Sentinel-2 or Google Earth Engine
- Real traffic data from city APIs or transportation departments

---
