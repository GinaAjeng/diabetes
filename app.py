import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib # For saving/loading the model
import os # For checking if a file exists

# Set page configuration for better layout
st.set_page_config(layout="wide", page_title="Diabetes Prediction Dashboard")

# Define the filename for the saved model
MODEL_FILENAME = 'random_forest_model.joblib'

# --- Helper function to generate dummy data (for demonstration) ---
# In a real application, you would load your actual dataset.
DATA_FILENAME = 'diabetes.csv' # <<< IMPORTANT: Change this to your actual file name!

@st.cache_data # Keep this decorator, it makes loading fast after the first time
def load_data():
    """Loads the actual dataset from a CSV file."""
    try:
        df = pd.read_csv(DATA_FILENAME)
        st.success(f"Successfully loaded '{DATA_FILENAME}'!")
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{DATA_FILENAME}' not found. "
                 f"Please make sure '{DATA_FILENAME}' is in the same directory as this app.")
        st.stop() # Stop the app if the dataset isn't found
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        st.stop() # Stop the app if there's another loading error


# --- Model Training/Loading Function (cached) ---
@st.cache_resource # Use st.cache_resource for models/objects that should not be re-run on every interaction
def get_model(df):
    """
    Loads a pre-trained Random Forest Classifier model if it exists,
    otherwise trains a new one and saves it.
    """
    model = None
    metrics = None
    X_test = None
    y_test = None

    if os.path.exists(MODEL_FILENAME):
        st.info(f"Loading model from '{MODEL_FILENAME}'...")
        try:
            model = joblib.load(MODEL_FILENAME)
            st.success("Model loaded successfully!")

            # To provide metrics for the loaded model, we need to re-evaluate it
            # on the test set derived from the current data.
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            y_pred = model.predict(X_test)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            st.info("Metrics re-calculated for the loaded model.")

        except Exception as e:
            st.error(f"Error loading model from '{MODEL_FILENAME}': {e}. Training a new model instead.")
            # Fallback to training if loading fails
            model, metrics, X_test, y_test = _train_and_save_model(df)
    else:
        st.warning(f"Model file '{MODEL_FILENAME}' not found. Training a new model...")
        model, metrics, X_test, y_test = _train_and_save_model(df)

    return model, metrics, X_test, y_test

def _train_and_save_model(df):
    """Helper function to train and save the model."""
    st.write("Training model... This might take a moment.")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    st.success("Model training complete!")

    # Save the newly trained model
    try:
        joblib.dump(model, MODEL_FILENAME)
        st.success(f"Model saved to '{MODEL_FILENAME}' for future use.")
    except Exception as e:
        st.error(f"Could not save model to '{MODEL_FILENAME}': {e}")

    return model, metrics, X_test, y_test


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1. Exploratory Data Analysis", "2. Model Evaluation", "3. Prediction"])

# Load the data once
df = load_data()

# --- Page 1: Exploratory Data Analysis (EDA) ---
if page == "1. Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Dive into the dataset to understand its structure, distributions, and relationships.")

    st.header("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    st.write("Dataset Information:")
    # Using a string buffer to capture info() output
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("Descriptive Statistics:")
    st.dataframe(df.describe())

    st.header("Distribution of Features")
    selected_feature_dist = st.selectbox("Select a feature to view its distribution:", df.columns[:-1]) # Exclude 'Outcome'

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_feature_dist], kde=True, ax=ax, color='skyblue')
    ax.set_title(f'Distribution of {selected_feature_dist}')
    ax.set_xlabel(selected_feature_dist)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

    st.header("Correlation Heatmap")
    st.write("Understand the correlation between different features.")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Correlation Matrix of Features')
    st.pyplot(fig)
    plt.close(fig)

    st.header("Outcome Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Outcome', data=df, ax=ax, palette='viridis')
    ax.set_title('Distribution of Diabetes Outcome')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'])
    st.pyplot(fig)
    plt.close(fig)


# --- Page 2: Model Evaluation ---
elif page == "2. Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("Assess the performance of the trained Random Forest Classifier.")

    st.info("The model will be loaded from file (if available) or trained when you navigate to this page.")
    model, metrics, X_test, y_test = get_model(df) # Get/load the model

    if model and metrics: # Ensure model and metrics are available
        st.header("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        col2.metric("Precision", f"{metrics['precision']:.2f}")
        col3.metric("Recall", f"{metrics['recall']:.2f}")
        col4.metric("F1-Score", f"{metrics['f1']:.2f}")

        st.header("Confusion Matrix")
        st.write("A confusion matrix shows the number of correct and incorrect predictions made by the classification model compared to the actual outcomes.")

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion_matrix'], display_labels=['No Diabetes', 'Diabetes'])
        disp.plot(cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        plt.close(fig)

        st.header("Feature Importances")
        st.write("Understand which features contributed most to the model's predictions.")
        feature_importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax, palette='coolwarm')
        ax.set_title('Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Model could not be loaded or trained. Please check for errors.")


# --- Page 3: Prediction ---
elif page == "3. Prediction":
    st.title("🔮 Make a Diabetes Prediction")
    st.markdown("Enter the patient's details to get a diabetes prediction.")

    st.info("The model will be loaded from file (if available) or trained for prediction.")
    model, _, _, _ = get_model(df) # Ensure model is trained/loaded

    if model: # Only proceed if model is successfully loaded or trained
        st.header("Patient Input Features")

        # Input fields for each feature
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1, help="Number of times pregnant")
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=122, value=70, help="Diastolic blood pressure")
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20, help="Triceps skin fold thickness")
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, value=79, help="2-Hour serum insulin")
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=67.1, value=25.0, step=0.1, help="Body Mass Index")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.471, step=0.001, format="%.3f", help="A function which scores likelihood of diabetes based on family history")
            age = st.number_input("Age (years)", min_value=21, max_value=81, value=30, help="Age in years")

        # Create a DataFrame from user input
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        st.subheader("Input Data Summary:")
        st.dataframe(input_data)

        if st.button("Predict Diabetes"):
            try:
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.error(f"**Prediction: The patient is likely to have Diabetes.** (Probability: {prediction_proba[1]*100:.2f}%)")
                else:
                    st.success(f"**Prediction: The patient is likely NOT to have Diabetes.** (Probability: {prediction_proba[0]*100:.2f}%)")

                st.markdown("---")
                st.info("Disclaimer: This prediction is based on a machine learning model trained on a sample dataset and should not be used as medical advice. Consult a healthcare professional for diagnosis.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure all input fields are filled correctly.")
    else:
        st.error("Model is not available for prediction. Please check the 'Model Evaluation' page for training status.")

