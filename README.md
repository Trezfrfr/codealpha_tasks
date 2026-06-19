<div align="center"><h1>Code Alpha Tasks</h1></div>
<br>


- ## Task1: Iris Species Predictor Pro

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

- ## Task2- car price prediction

An interactive, full-stack web application designed for the used car market that provides business intelligence visualizations and machine learning-powered resale value predictions.

---

## Project Description

This application acts as a dual-purpose platform: providing a Business Intelligence (BI) Dashboard to visualize historical pricing trends and depreciation metrics, and an Artificial Intelligence (AI) Valuation Tool that uses machine learning to predict the current and future resale value of vehicles based on their specific characteristics.

## How the Prediction is Made

The core of this application is a robust Machine Learning pipeline built using Scikit-Learn. Instead of just passing raw data to a model, the system uses a professional `Pipeline` and `ColumnTransformer` architecture to ensure data integrity and prevent data leakage.

* **Preprocessing Categorical Data:** Text columns like `Fuel_Type`, `Transmission`, and `Selling_type` are passed through a `SimpleImputer` (to fill any missing values with the most frequent category) and then converted into a binary matrix using `OneHotEncoder`.
* **Preprocessing Numerical Data:** Number columns are imputed using the median to handle missing data, and then passed through a `StandardScaler` to normalize the ranges, ensuring large numbers (like `Driven_kms`) don't overwhelm smaller numbers (like `Age`).
* **The Algorithm:** The preprocessed data is fed into a **Random Forest Regressor**. This ensemble learning algorithm builds 100 decision trees to learn the complex, non-linear relationships between a car's features and its final selling price (e.g., how a specific brand mitigates the steepness of a depreciation curve).
* **Evaluation:** The model is evaluated on a 20% holdout test set using the R² Score to ensure high predictive accuracy.

## Added Features

To squeeze the maximum predictive power out of the raw `car_data.csv` dataset, the application performs dynamic Feature Engineering under the hood:

* **Brand Extraction:** Extracted from the first word of the `Car_Name` to capture brand prestige and goodwill.
* **Dynamic Age Calculation:** Calculated dynamically by subtracting the car's manufacturing `Year` from the current year.
* **Usage Intensity (`Kms_Per_Year`):** Calculated by dividing total `Driven_kms` by the `Age` (plus 1 to avoid division by zero). This helps the model differentiate between gently used cars and heavily abused fleet vehicles.
* **Ownership Flag (`Is_First_Owner`):** Converts the integer-based ownership column into a strict binary flag to capture the premium value of single-owner vehicles.

## Frontend Functionality

The frontend is a highly interactive, reactive web interface built with Streamlit. It allows users to interact with complex data and machine learning models without writing a single line of code. The dashboard is divided into 5 distinct analytical modes:

### Price Comparison Across Years
* Renders a visual line chart tracking average selling prices across different manufacturing years.

### Filter Cars According to Needs
* A digital showroom that dynamically filters the raw dataset based on user-selected transmission preferences and odometer cutoffs.

### Price Depreciation Analysis
* Computes and visualizes the exact financial drop between a car's original showroom price and its current resale value, highlighting the top 20 most depreciated assets.

### Price Predictor Model
* A real-time appraisal form. Users input vehicle specs, and the frontend instantly passes the data through the ML pipeline to output an estimated market price.

### Future Resale Value Forecaster
* A simulation engine that takes current car specs and estimated annual driving distance to project a 5-year predictive depreciation curve into the future.

## Tools & Libraries Used

* **Python 3:** The core programming language.
* **Streamlit:** Used to build the interactive web frontend and manage application state without requiring HTML/CSS/JS.
* **Scikit-Learn:** Powered the Machine Learning pipeline, Random Forest model, and calculated performance metrics.
* **Pandas:** Handled data ingestion (CSV reading), manipulation, and dynamic feature engineering.
* **NumPy:** Handled high-performance numerical operations.
* **Matplotlib:** Generated the data visualizations, charts, and predictive depreciation curves.
