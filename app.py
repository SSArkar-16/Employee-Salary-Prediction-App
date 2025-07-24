import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="📊 Income Prediction App", layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center; color: teal;'>💼 Income Prediction using Machine Learning Algorithms</h1>", unsafe_allow_html=True)
st.write("This app allows you to explore multiple ML models and predict whether a person's income exceeds $50K/year.")

# ---------------------------
# Load & Prepare Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    df.replace('?', 'Others', inplace=True)
    df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
    df.drop(columns=['education'], inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    df = df[(df['age'] >= 17) & (df['age'] <= 75)]
    df = df[(df['educational-num'] >= 5) & (df['educational-num'] <= 16)]
    df = df[(df['hours-per-week'] >= 8) & (df['hours-per-week'] <= 80)]
    df = df[(df['capital-gain'] <= 50000)]
    df = df[(df['capital-loss'] <= 4000)]

    return df

@st.cache_resource
def preprocess_and_train(df):
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        trained[name] = {
            "model": model,
            "accuracy": acc,
            "report": report,
            "cm": cm
        }

    return X_train, X_test, y_train, y_test, scaler, trained

# ---------------------------
# Load data and train models
# ---------------------------
df = load_data()
X_train, X_test, y_train, y_test, scaler, trained_models = preprocess_and_train(df)

# ---------------------------
# Sidebar Settings
# ---------------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Select Model", list(trained_models.keys()))

model_info = trained_models[model_name]

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📈 Model Performance", "🧪 Predict Income"])

# ---------------------------
# Tab 1 - Model Performance
# ---------------------------
with tab1:
    st.markdown("### 🔍 Model Evaluation")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("🔢 Accuracy", f"{model_info['accuracy']*100:.2f}%")

        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(model_info['report']).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)

    with col2:
        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(model_info["cm"], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ---------------------------
# Tab 2 - Prediction
# ---------------------------
with tab2:
    st.markdown("### 🧪 Try Your Own Input")

    # Only these 4 features used
    st.info("This prediction is based only on **age**, **occupation**, **sex**, and **education**.")

    # Load full set of encoders
    global occupation_encoder, sex_encoder, education_encoder

    # Prepare dropdowns using original class labels
    age = st.number_input("age", min_value=17, max_value=75, value=30)

    selected_occupation = st.selectbox("occupation", occupation_encoder.classes_)
    selected_sex = st.selectbox("sex", df['sex'].sort_values().unique())
    selected_education = st.selectbox("education", df['education'].sort_values().unique())

    # Encode user input
    occ_encoded = occupation_encoder.transform([selected_occupation])[0]

    sex_le = LabelEncoder()
    sex_le.fit(df['sex'])
    sex_encoded = sex_le.transform([selected_sex])[0]

    edu_le = LabelEncoder()
    edu_le.fit(df['education'])
    edu_encoded = edu_le.transform([selected_education])[0]

    # Construct full feature vector with placeholder values for other columns
    feature_dict = {
        'age': age,
        'workclass': 0,
        'fnlwgt': 0,
        'educational-num': edu_encoded,
        'marital-status': 0,
        'occupation': occ_encoded,
        'relationship': 0,
        'race': 0,
        'sex': sex_encoded,
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 0
    }

    input_df = pd.DataFrame([feature_dict])

    # Apply the same scaler
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model_info['model'].predict(input_scaled)[0]

    # Show result
    st.success(f"✅ Prediction: {'>50K' if prediction == 1 else '<=50K'}")
    st.info(f"👤 Age: {age} | 👔 Occupation: {selected_occupation} | 🎓 Education: {selected_education} | ⚧️ Sex: {selected_sex}")

    st.warning("⚠️ This prediction is an estimate based on historical data and may not reflect real-life outcomes accurately. Please use it for educational purposes only.")


# ---------------------------
# Footer
# ---------------------------
st.markdown("""---""", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center;'>
        <p>Built by <strong>Saranya Sarkar </strong>🚀</p>
        <p>Connect with me:</p>
        <a href="https://www.linkedin.com/in/saranya-sarkar/" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" 
                 alt="LinkedIn" width="30" style="margin-right:10px;" />
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

