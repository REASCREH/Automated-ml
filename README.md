# Automated Machine Learning Classification and Regression Application


Introducing an Automated Machine Learning (AutoML) Classification and Regression App designed to streamline the machine learning workflow for both novices and experienced practitioners. This innovative application empowers users to effortlessly manage data, train models, and evaluate performance—all through an intuitive interface.




## Key Features


Upload Datasets: Support for CSV, Excel, or ZIP formats.
Exploratory Data Analysis (EDA): Automated insights such as data types, missing values, and descriptive statistics.

Handle Missing Values: Automatically remove columns with high missing data and impute remaining values based on column type.

Encoding Methods: Choose from Label Encoding, One-Hot Encoding, or Target Encoding for categorical columns.
Model Selection: Select between classification and regression models.

Supported Algorithms:

Classification: XGBoost, LightGBM, CatBoost, Logistic Regression, SVM.

Regression: XGBoost, LightGBM, CatBoost, Linear Regression, SVR.

Model Training Options:

Default settings or custom parameters.

Bayesian Optimization for hyperparameter tuning.

Performance Evaluation: Metrics like Accuracy, Precision, Recall, F1-Score (for classification), and MSE, MAE (for regression).

Test Data Predictions: Upload separate test datasets for further predictions.


## Get Started Online


You can access the Automated Machine Learning Classification and Regression App online at the following link:

https://automated-ml.streamlit.app/

Instructions for Online Use
Visit the App: Open your web browser and go to https://automated-ml.streamlit.app/.

Upload Your Dataset:

Click the "Upload" button to select your dataset in CSV, Excel, or ZIP format.

The app will automatically process and display the first few rows of your dataset.

Explore the Data:

1-Review the automated exploratory data analysis (EDA) provided by the app, which includes information about data types, missing values, and descriptive statistics.
Preprocess the Data:

2-The app will handle missing values and mixed-type columns automatically.
3-Choose an encoding method for categorical columns based on your requirements.
Select a Model:

4-Choose whether you want to perform classification or regression.
5-Select the desired machine learning algorithm from the available options.

5-Train Your Model:

Decide whether to use default parameters, customize key parameters, or perform Bayesian optimization for hyperparameter tuning.

6-Evaluate Model Performance:

After training, review the model performance metrics provided by the app, including classification or regression metrics.

7-Make Predictions:

Upload a separate test dataset for further predictions.
Review the predictions and download the results if needed.


## To run this application locally, follow these steps:



Clone the project

```bash
  git clone https://github.com/REASCREH/Automated-ml.git


```

Go to the project directory

```bash
  cd machine-learning-app

```

Install dependencies

```bash
  pip install -r requirements.txt

```

Start the server

```bash
  streamlit run app3.py

```


## Benefits of Using This App

User-Friendly Interface: Streamlit provides an intuitive, interactive interface for users, making complex machine learning processes accessible.

Comprehensive Data Handling: The app efficiently manages data uploads, preprocessing, and exploration, which saves time for users.

Flexible Model Selection: Users can choose from multiple algorithms tailored to their specific needs, whether for classification or regression tasks.

Automated Insights: The automated EDA feature helps users quickly understand their data without extensive statistical knowledge.

Enhanced Performance through Tuning: Bayesian Optimization allows users to maximize model performance without needing in-depth knowledge of hyperparameter settings.

Predictive Capabilities: Users can easily apply their trained models to new data, enabling real-time decision-making.

Educational Resource: Ideal for both beginners and advanced users to learn about machine learning workflows and model evaluations.

