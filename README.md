# Car Market Intelligence & Predictive Engine

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
