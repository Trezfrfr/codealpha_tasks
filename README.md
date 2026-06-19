<div align="center"><h1>Code Alpha Tasks</h1></div>
<br>


## Task1: Iris Species Predictor Pro

An interactive machine learning web application that predicts the species of an Iris flower (*Setosa, Versicolor, or Virginica*) based on its physical measurements. This project features a robust Scikit-Learn machine learning pipeline and a responsive user interface built entirely in Python.

---

## Project Description

This application serves as an end-to-end demonstration of a machine learning workflow. It takes raw botanical measurements, automatically engineers complex features, makes a real-time prediction using a trained Random Forest model, and provides a comprehensive breakdown of the model's performance on unseen testing data.

## How the Prediction is Made

The prediction logic relies on a strictly defined machine learning pipeline designed to prevent data leakage and maximize accuracy.

* **Base Measurements:** The model accepts four continuous numerical inputs representing the physical dimensions (in centimeters) of the flower: Sepal Length, Sepal Width, Petal Length, and Petal Width.
* **Automated Feature Engineering:** Before reaching the model, the inputs pass through a `PolynomialFeatures` transformer. This step automatically generates interaction terms (e.g., multiplying length by width to approximate the **Area** of the petals and sepals). These engineered proportions provide highly distinct signals for classification.
* **The Algorithm:** The core predictor is a **Random Forest Classifier** trained with 100 decision trees. It processes the scaled base measurements alongside the engineered features to output a highly confident species classification.

## Added Features

* **Zero Data Leakage Pipeline:** Imputation, feature engineering, and standardization (`StandardScaler`) are bundled directly into a Scikit-Learn `Pipeline`.
* **Dynamic Target Encoding:** The text-based species labels are automatically transformed into numerical values during training and safely reverted to human-readable text for the final output using `LabelEncoder`.
* **Model Evaluation:** The dataset is split (80/20) to ensure the model is evaluated on data it has never seen before, proving its real-world reliability.

## Frontend Functionality

The user interface is split into two intuitive tabs:

### Live Prediction Tab
* Features an interactive dashboard with sliders to input custom flower dimensions.
* Displays the selected measurements in clean metric cards.
* Outputs the predicted species in a highly visible success banner.
* Includes a dynamic bar chart illustrating the model's confidence probability across all three possible species.

### Model Performance Tab
* Displays the model's overall accuracy score on the 20% testing split.
* Renders a visual **Confusion Matrix** (via Seaborn) to show exactly where the model succeeded and where it confused similar species.
* Provides a detailed **Classification Report** in a formatted data table, breaking down precision, recall, and f1-scores.

## Tools & Libraries Used

* **Python 3:** The core programming language.
* **Streamlit:** Used to build the interactive web frontend and dashboard components without requiring HTML/CSS/JS.
* **Scikit-Learn:** Powered the machine learning model, data preprocessing, feature engineering, and performance metrics.
* **Pandas:** Handled data structures and organized the inputs for the model and visual reports.
* **Matplotlib & Seaborn:** Generated the data visualizations, specifically the heatmap for the confusion matrix.
