import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Function to Load Data
@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv('online_payment_fraud.csv')

# Function to Train and Save the Model
def train_and_evaluate_model(df):
    # Preprocessing and splitting
    X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
    y = df['isFraud']

    categorical_features = ['type']
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    X_preprocessed = preprocessor.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_preprocessed, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    # Train the model
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    joblib.dump(pipeline, 'fraud_model.pkl')

    # Predictions and evaluation
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)  # Fixed this line
    confusion = confusion_matrix(y_test, y_pred)

    return accuracy, report, confusion



# Function to Load the Trained Model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('fraud_model.pkl')

# Streamlit Web Interface
st.title("Online Payment Fraud Detection")

# Step 1: Load Data
df = load_data()
st.subheader("Dataset Preview")
st.write(df.head())

# Step 2: Visualize Data
st.write("### Fraud Distribution")
fig, ax = plt.subplots()
sns.countplot(x='isFraud', data=df, ax=ax)
ax.set_title("Fraud vs Non-Fraud Transactions")
st.pyplot(fig)

# Step 3: Train the Model
st.subheader("Model Training")
if st.button("Train Model"):
    with st.spinner("Training model..."):
        accuracy, report, confusion = train_and_evaluate_model(df)
    st.success(f"Model Trained Successfully! Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(report)
    st.text("Confusion Matrix:")
    st.write(confusion)

# Step 4: Load the Trained Model
model = None
try:
    model = load_model()
except:
    st.warning("Please train the model first by clicking the 'Train Model' button.")

# Step 5: Make Predictions
st.subheader("Make a Prediction")
if model:
    # Collect user input for prediction
    step = st.number_input('Step', min_value=1, max_value=744, value=100)
    txn_type = st.selectbox('Transaction Type', df['type'].unique())
    amount = st.number_input('Amount', min_value=0.0, value=100.0)
    oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0, value=100000.0)
    newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0, value=90000.0)
    oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0, value=150000.0)
    newbalanceDest = st.number_input('New Balance Destination', min_value=0.0, value=160000.0)

    if st.button("Predict Fraud"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'step': [step],
            'type': [txn_type],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest]
        })

        try:
            # Use the trained pipeline to transform and predict
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1]

            # Display result
            if prediction == 1:
                st.error(f"**Fraudulent Transaction!** (Probability: {prediction_proba:.2f})")
            else:
                st.success(f"**Legitimate Transaction.** (Probability: {1 - prediction_proba:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
