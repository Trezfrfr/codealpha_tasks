<div align="center"><h1>Code Alpha Tasks</h1></div>
<br>

## Project Guidelines & Repository Structure

This repository contains multiple independent machine learning tasks. To keep the codebase organized, the projects are divided among specific feature branches. 

Please switch to the corresponding branch to view the code and documentation for each task:

* **Task 1:** `Task1iris-flower-classification` branch
* **Task 2:** `Task2car-sales-analysis` branch

<br>

## ⚙️ Setup & Installation Guide

Both projects in this repository are built using Python and Streamlit. To run either project locally, follow these complete, step-by-step instructions.

1. ## Clone the Repository
 First, download the project files to your local machine and navigate into the project folder.

2. ## Switch to the Desired Project Branch
Since the tasks are divided into branches, switch to the specific project you want to run:

Bash<br>
To run Task 1:
git checkout iris-classification
<br>
OR,to run Task 2:
git checkout car-sales-analysis

3. ## Create a Virtual Environment (Recommended)
It is best practice to create a virtual environment to prevent dependency conflicts.

For Windows:

Bash <br> 
python -m venv venv <br>
venv\Scripts\activate <br><br>

For macOS and Linux:

Bash <br>
python3 -m venv venv <br>
source venv/bin/activate

4. ## Install Required Dependencies
With your virtual environment activated, install the necessary Python libraries. Ensure you have a requirements.txt file in the branch, then run:
<br>
Bash<br>
pip install -r requirements.txt<br>
(Note: If you do not have a requirements file generated yet, you can manually install the core stack for both projects by running: pip install streamlit pandas numpy scikit-learn matplotlib seaborn)

5. ## Launch the Application
Finally, start the Streamlit server:

Bash
streamlit run app.py
(Note: Replace app.py if your main Python script has a different name).

Once executed, Streamlit will automatically open a new tab in your default web browser hosting the interactive dashboard (usually accessible at http://localhost:8501).

<br>


- ## Task1:Iris Flower Intelligence & Predictive Engine

An interactive Streamlit web application that provides exploratory data visualization and machine learning-powered species prediction for the Iris dataset.

---

1. ## Project Description

This dual-purpose application serves as both a data exploration dashboard and a real-time classification tool. It allows users to visually analyze the physical characteristics of Iris flowers through interactive charts, and utilizes a trained machine learning pipeline to instantly classify new flowers into their respective species based on sepal and petal dimensions. 

2. ## How the Prediction is Made

The prediction logic relies on a strictly defined machine learning pipeline designed to prevent data leakage and maximize classification accuracy.

* **Data Scaling:** The continuous numerical inputs (Sepal Length, Sepal Width, Petal Length, and Petal Width) are passed through a `StandardScaler` to normalize the data distributions and ensure all features contribute equally to the model.
* **The Algorithm:** The standardized data is fed into a **Random Forest Classifier**. This ensemble algorithm builds 100 decision trees (`n_estimators=100`) to learn the precise dimensional thresholds that separate the different Iris species.
* **Pipeline Integrity & Evaluation:** The `StandardScaler` and `RandomForestClassifier` are bundled directly into a Scikit-Learn `Pipeline`. The model's structural accuracy is evaluated on an isolated 20% holdout test set to ensure real-world reliability.

3. ## Added Features

* **Automated Data Cleansing:** The application dynamically sanitizes the raw `iris.csv` file upon loading by standardizing column names (lowercasing, replacing spaces/dots with underscores), stripping redundant 'id' columns, and homogenizing target variables (converting variations like 'variety' or 'class' into a standard 'species' column).
* **High-Performance Caching:** Utilizes Streamlit's native `@st.cache_data` and `@st.cache_resource` decorators. This ensures the raw dataset is loaded and the machine learning model is trained only once per session, resulting in a lightning-fast, reactive user experience.

4. ## Frontend Functionality

The frontend is a highly interactive interface built with Streamlit, divided into four distinct analytical modes:

### Dataset Overview & Averages
* Computes and displays an aggregated data table showing the average dimensions for each species.
* Renders a visual bar chart comparing the average petal lengths across the different flower types.

### Filter by Species
* Acts as a digital dataset explorer, allowing users to dynamically filter and view the raw data records based on a specific, selected species.

### Feature Scatter Analysis
* An interactive 2D visualization tool where users can dynamically select which physical features to map to the X and Y axes, making it easy to visually identify how different species cluster together.

### Real-time Species Predictor
* Displays the core model's accuracy metric based on the testing split.
* Features a live appraisal form with interactive sliders for all four botanical dimensions. When submitted, the data is passed through the ML pipeline to output a highly visible predicted species classification.

5. ## Tools & Libraries Used

* **Python 3:** The core programming language.
* **Streamlit:** Used to build the interactive web frontend, analytical modes, and state management without HTML/CSS/JS.
* **Scikit-Learn:** Powered the machine learning `Pipeline`, data scaling, `RandomForestClassifier`, and accuracy metrics.
* **Pandas:** Handled data ingestion, dynamic column cleansing, grouping, and feature payload structuring.
* **NumPy:** Supported underlying high-performance numerical operations.
* **Matplotlib:** Generated the data visualizations, including the comparative bar charts and interactive scatter plots.
<br>
<br>

- ## Task2- car price prediction

An interactive, full-stack web application designed for the used car market that provides business intelligence visualizations and machine learning-powered resale value predictions.

---

1. ## Project Description

This application acts as a dual-purpose platform: providing a Business Intelligence (BI) Dashboard to visualize historical pricing trends and depreciation metrics, and an Artificial Intelligence (AI) Valuation Tool that uses machine learning to predict the current and future resale value of vehicles based on their specific characteristics.

2. ## How the Prediction is Made

The core of this application is a robust Machine Learning pipeline built using Scikit-Learn. Instead of just passing raw data to a model, the system uses a professional `Pipeline` and `ColumnTransformer` architecture to ensure data integrity and prevent data leakage.

* **Preprocessing Categorical Data:** Text columns like `Fuel_Type`, `Transmission`, and `Selling_type` are passed through a `SimpleImputer` (to fill any missing values with the most frequent category) and then converted into a binary matrix using `OneHotEncoder`.
* **Preprocessing Numerical Data:** Number columns are imputed using the median to handle missing data, and then passed through a `StandardScaler` to normalize the ranges, ensuring large numbers (like `Driven_kms`) don't overwhelm smaller numbers (like `Age`).
* **The Algorithm:** The preprocessed data is fed into a **Random Forest Regressor**. This ensemble learning algorithm builds 100 decision trees to learn the complex, non-linear relationships between a car's features and its final selling price (e.g., how a specific brand mitigates the steepness of a depreciation curve).
* **Evaluation:** The model is evaluated on a 20% holdout test set using the R² Score to ensure high predictive accuracy.

3. ## Added Features

To squeeze the maximum predictive power out of the raw `car_data.csv` dataset, the application performs dynamic Feature Engineering under the hood:

* **Brand Extraction:** Extracted from the first word of the `Car_Name` to capture brand prestige and goodwill.
* **Dynamic Age Calculation:** Calculated dynamically by subtracting the car's manufacturing `Year` from the current year.
* **Usage Intensity (`Kms_Per_Year`):** Calculated by dividing total `Driven_kms` by the `Age` (plus 1 to avoid division by zero). This helps the model differentiate between gently used cars and heavily abused fleet vehicles.
* **Ownership Flag (`Is_First_Owner`):** Converts the integer-based ownership column into a strict binary flag to capture the premium value of single-owner vehicles.

4. ## Frontend Functionality

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

5. ## Tools & Libraries Used

* **Python 3:** The core programming language.
* **Streamlit:** Used to build the interactive web frontend and manage application state without requiring HTML/CSS/JS.
* **Scikit-Learn:** Powered the Machine Learning pipeline, Random Forest model, and calculated performance metrics.
* **Pandas:** Handled data ingestion (CSV reading), manipulation, and dynamic feature engineering.
* **NumPy:** Handled high-performance numerical operations.
* **Matplotlib:** Generated the data visualizations, charts, and predictive depreciation curves.
