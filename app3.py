import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
import numpy as np
import zipfile
import tempfile

# Upload dataset
st.title("Machine Learning App for Classification and Regression")
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, ZIP)", type=["csv", "xlsx", "xls", "zip"])

if uploaded_file:
    if uploaded_file.name.endswith('.zip'):
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                file_list = z.namelist()
                for filename in file_list:
                    if filename.endswith('.csv'):
                        with z.open(filename) as csv_file:
                            try:
                                df = pd.read_csv(csv_file, encoding='utf-8')
                            except UnicodeDecodeError:
                                st.warning("UTF-8 encoding failed, trying ISO-8859-1 encoding.")
                                df = pd.read_csv(csv_file, encoding='ISO-8859-1')
                            st.success("File successfully uploaded!")
                            st.write(df.head())
                            break
        except Exception as e:
            st.error(f"Error reading ZIP file: {e}")
    else:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.success("File successfully uploaded!")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Step 1: Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Shape of the dataset:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write("Descriptive Statistics:")
    st.write(df.describe())

    # Step 2: Handling Missing Values
    st.subheader("Handling Missing Values")
    st.write("Removing columns with more than 60% missing values.")
    missing_threshold = 0.6
    df = df.loc[:, df.isnull().mean() < missing_threshold]
    st.write("Columns after removal:", df.columns)

    st.write("Imputing missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    st.write("Missing values handled!")

    # Step 3: Removing Mixed Type Columns
    st.subheader("Removing Mixed Type Columns")
    mixed_type_columns = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str)).any() and df[col].apply(lambda x: isinstance(x, (int, float))).any():
            mixed_type_columns.append(col)

    df.drop(columns=mixed_type_columns, inplace=True)
    st.write("Removed columns with mixed data types:", mixed_type_columns)

    # Step 4: Encoding Categorical Variables One-by-One
    st.subheader("Encoding Categorical Variables (One-by-One)")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    encoders = {}
    for col in categorical_cols:
        st.write(f"Encoding column: {col}")
        encoding_method = st.selectbox(f"Choose encoding method for '{col}':",
                                       ["Label Encoding", "One-Hot Encoding", "Target Encoding"], key=col)

        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            st.write(f"Applied Label Encoding to {col}")

        elif encoding_method == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=[col])
            st.write(f"Applied One-Hot Encoding to {col}")

        elif encoding_method == "Target Encoding":
            target_col = st.selectbox(f"Select target column for Target Encoding for {col}:", df.columns)
            means = df.groupby(col)[target_col].mean()
            df[col] = df[col].map(means)
            st.write(f"Applied Target Encoding to {col}")

    # Step 5: Select Target Column
    st.subheader("Select Target Column")
    target_col = st.selectbox("Choose your target column:", df.columns)

    # Step 6: Train-Test Split
    test_size = st.slider("Test Size (percentage)", 10, 50, 20) / 100
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Step 7: Model Selection
    st.subheader("Model Selection")
    problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])

    if problem_type == "Classification":
        model_choice = st.selectbox("Choose Classification Model", ["XGBoost", "LightGBM", "CatBoost", "Logistic Regression", "SVM"])
        if model_choice == "XGBoost":
            model = XGBClassifier()
        elif model_choice == "LightGBM":
            model = LGBMClassifier()
        elif model_choice == "CatBoost":
            model = CatBoostClassifier(verbose=0)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "SVM":
            model = SVC()

    elif problem_type == "Regression":
        model_choice = st.selectbox("Choose Regression Model", ["XGBoost", "LightGBM", "CatBoost", "Linear Regression", "SVM"])
        if model_choice == "XGBoost":
            model = XGBRegressor()
        elif model_choice == "LightGBM":
            model = LGBMRegressor()
        elif model_choice == "CatBoost":
            model = CatBoostRegressor(verbose=0)
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "SVM":
            model = SVR()

    # Step 8: Model Training
    st.subheader("Model Training")
    tuning_option = st.radio("Apply Model with or without Hyperparameter Tuning", ["Without Parameters", "Custom Parameters", "Bayesian Optimization"])

    if tuning_option == "Without Parameters":
        model.fit(X_train, y_train)
    elif tuning_option == "Custom Parameters":
        st.write("Enter custom parameters below:")

        # Parameters (specific to each model)
        if model_choice in ["XGBoost", "LightGBM", "CatBoost"]:
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
            n_estimators = st.slider("n_estimators", 50, 500, 100)
            max_depth = st.slider("max_depth", 3, 10, 6)
            model.set_params(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

        model.fit(X_train, y_train)
    elif tuning_option == "Bayesian Optimization":
        st.write("Select parameters for Bayesian Optimization")

        def bayesian_opt_function(learning_rate, n_estimators, max_depth):
            params = {
                "learning_rate": learning_rate,
                "n_estimators": int(n_estimators),
                "max_depth": int(max_depth)
            }
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        if model_choice in ["XGBoost", "LightGBM", "CatBoost"]:
            optimizer = BayesianOptimization(
                f=bayesian_opt_function,
                pbounds={
                    "learning_rate": (0.01, 0.5),
                    "n_estimators": (50, 500),
                    "max_depth": (3, 10),
                },
                random_state=42,
            )
            optimizer.maximize(init_points=5, n_iter=10)

            best_params = optimizer.max['params']
            model.set_params(
                learning_rate=best_params['learning_rate'],
                n_estimators=int(best_params['n_estimators']),
                max_depth=int(best_params['max_depth'])
            )

        model.fit(X_train, y_train)

    # Step 9: Model Evaluation
    st.subheader("Model Evaluation")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if problem_type == "Classification":
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        st.write(f"Training Accuracy: {train_accuracy}")
        st.write(f"Testing Accuracy: {test_accuracy}")

        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')

        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")

    elif problem_type == "Regression":
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        st.write(f"Training MSE: {train_mse}")
        st.write(f"Testing MSE: {test_mse}")
        st.write(f"Training MAE: {train_mae}")
        st.write(f"Testing MAE: {test_mae}")

    # Step 10: Upload Test Data for Further Evaluation
    st.subheader("Upload Test Data for Further Evaluation")
    test_file = st.file_uploader("Upload test data (CSV, Excel)", type=["csv", "xlsx", "xls"], key="test_file")
    
    if test_file:
        try:
            if test_file.name.endswith('.csv'):
                test_df = pd.read_csv(test_file)
            elif test_file.name.endswith('.xlsx') or test_file.name.endswith('.xls'):
                test_df = pd.read_excel(test_file)
            
            st.write("Test Data Preview:")
            st.write(test_df.head())
            
            # Make predictions on the uploaded test data
            X_new_test = test_df.drop(target_col, axis=1)
            y_new_test_pred = model.predict(X_new_test)

            st.write("Predictions on uploaded test data:")
            st.write(pd.DataFrame(y_new_test_pred, columns=["Predictions"]))
            
        except Exception as e:
            st.error(f"Error reading test file: {e}")
