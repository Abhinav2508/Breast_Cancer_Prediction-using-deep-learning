## ✨ Author

**Abhinav Kumar**  
- 🔗 [LinkedIn](https://www.linkedin.com/in/abhinav-kumar-b0b0ba253/)  
- 💻 [GitHub](https://github.com/Abhinav2508)
  
---

#  Breast Cancer Classification using Deep Learning

This project is a Deep Learning-based binary classification model built to detect whether a tumor is **malignant** or **benign** using the **Breast Cancer Wisconsin Diagnostic Dataset** from `sklearn.datasets`. 

Achieving high accuracy with proper validation, the model demonstrates how neural networks can effectively classify medical data when properly preprocessed and trained.

---

## 📌 Project Overview

- ** Dataset**: Breast Cancer Wisconsin Diagnostic Data (from `sklearn.datasets`)
- ** Frameworks Used**: TensorFlow, Keras, Scikit-learn
- ** Objective**: Classify tumor as `Malignant (1)` or `Benign (0)`
- **⚙ Model Used**: Fully Connected Neural Network
- ** Final Accuracy**:
  - **Train Accuracy**: `99.56%`
  - **Test Accuracy**: `98.25%`
  - **Test Confusion Matrix**:
    ```
    [[41  2]
     [ 0 71]]
    ```

---

## 🛠 Tech Stack

- Python 🐍
- TensorFlow / Keras
- Scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn, Plotly

---

##  Project Structure

---

## 🚀 Workflow Summary

1. **Importing Libraries**
2. **Loading Dataset**
3. **Exploratory Data Analysis (EDA)**
4. **Data Preprocessing**
   - Standard Scaling using `StandardScaler`
   - Train-Test Split (80-20)
5. **Model Building**
   - Neural Network with 3 layers
   - Activation: ReLU, Sigmoid
   - Loss: Binary Crossentropy
   - Optimizer: Adam
6. **Model Training**
   - Epochs: 100
   - Validation Split: 0.2
7. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
8. **Prediction on Single Sample**

---

## 📌 Results Interpretation

- The model predicts with **very high confidence**, indicating it has learned patterns in the dataset effectively.
- The **confusion matrix shows zero false negatives**, which is critical in cancer detection.

---

## 📷 Visualizations

- Feature Distributions  
- Correlation Heatmap  
- Accuracy vs Epochs  
- Confusion Matrix

---

##  Acknowledgements

- **Dataset**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Libraries**: TensorFlow, Scikit-learn, Matplotlib, Seaborn

---





