# codealpha_tasks
<style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #24292e;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 1.5em;
        }
        h1 { 
            font-size: 2em; 
            border-bottom: 2px solid #eaecef; 
            margin-top: 0;
        }
        ul { padding-left: 20px; }
        li { margin-bottom: 0.5em; }
        code {
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
            font-size: 85%;
        }
        hr {
            height: 0.25em;
            padding: 0;
            margin: 24px 0;
            background-color: #e1e4e8;
            border: 0;
        }
    </style>
</head>
<body>
    <h1>🌿 Iris Species Predictor Pro</h1>
    <p>An interactive machine learning web application that predicts the species of an Iris flower (<em>Setosa, Versicolor, or Virginica</em>) based on its physical measurements. This project features a robust Scikit-Learn machine learning pipeline and a responsive user interface built entirely in Python.</p>

    <h2>📖 Project Description</h2>
    <p>This application serves as an end-to-end demonstration of a machine learning workflow. It takes raw botanical measurements, automatically engineers complex features, makes a real-time prediction using a trained Random Forest model, and provides a comprehensive breakdown of the model's performance on unseen testing data.</p>

    <h2>🧠 How the Prediction is Made</h2>
    <p>The prediction logic relies on a strictly defined machine learning pipeline designed to prevent data leakage and maximize accuracy.</p>
    <ul>
        <li><strong>Base Measurements:</strong> The model accepts four continuous numerical inputs representing the physical dimensions (in centimeters) of the flower: Sepal Length, Sepal Width, Petal Length, and Petal Width.</li>
        <li><strong>Automated Feature Engineering:</strong> Before reaching the model, the inputs pass through a <code>PolynomialFeatures</code> transformer. This step automatically generates interaction terms (e.g., multiplying length by width to approximate the <strong>Area</strong> of the petals and sepals). These engineered proportions provide highly distinct signals for classification.</li>
        <li><strong>The Algorithm:</strong> The core predictor is a <strong>Random Forest Classifier</strong> trained with 100 decision trees. It processes the scaled base measurements alongside the engineered features to output a highly confident species classification.</li>
    </ul>

    <h2>✨ Added Features</h2>
    <ul>
        <li><strong>Zero Data Leakage Pipeline:</strong> Imputation, feature engineering, and standardization (<code>StandardScaler</code>) are bundled directly into a Scikit-Learn <code>Pipeline</code>.</li>
        <li><strong>Dynamic Target Encoding:</strong> The text-based species labels are automatically transformed into numerical values during training and safely reverted to human-readable text for the final output using <code>LabelEncoder</code>.</li>
        <li><strong>Model Evaluation:</strong> The dataset is split (80/20) to ensure the model is evaluated on data it has never seen before, proving its real-world reliability.</li>
    </ul>

    <h2>🖥️ Frontend Functionality</h2>
    <p>The user interface is split into two intuitive tabs:</p>
    <ul>
        <li><strong>🎯 Live Prediction Tab:</strong>
            <ul>
                <li>Features an interactive dashboard with sliders to input custom flower dimensions.</li>
                <li>Displays the selected measurements in clean metric cards.</li>
                <li>Outputs the predicted species in a highly visible success banner.</li>
                <li>Includes a dynamic bar chart illustrating the model's confidence probability across all three possible species.</li>
            </ul>
        </li>
        <li><strong>📊 Model Performance Tab:</strong>
            <ul>
                <li>Displays the model's overall accuracy score on the 20% testing split.</li>
                <li>Renders a visual <strong>Confusion Matrix</strong> (via Seaborn) to show exactly where the model succeeded and where it confused similar species.</li>
                <li>Provides a detailed <strong>Classification Report</strong> in a formatted data table, breaking down precision, recall, and f1-scores.</li>
            </ul>
        </li>
    </ul>

    <h2>🛠️ Tools & Libraries Used</h2>
    <ul>
        <li><strong>Python 3:</strong> The core programming language.</li>
        <li><strong>Streamlit:</strong> Used to build the interactive web frontend and dashboard components without requiring HTML/CSS/JS.</li>
        <li><strong>Scikit-Learn:</strong> Powered the machine learning model, data preprocessing, feature engineering, and performance metrics.</li>
        <li><strong>Pandas:</strong> Handled data structures and organized the inputs for the model and visual reports.</li>
        <li><strong>Matplotlib &amp; Seaborn:</strong> Generated the data visualizations, specifically the heatmap for the confusion matrix.</li>
    </ul>
</body>
</html>
