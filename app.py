
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

HERE = Path(__file__).parent

@st.cache_resource
def load_model_files():
    model = joblib.load(HERE / "employee_attrition_model.pkl")
    features = joblib.load(HERE / "employee_attrition_features.pkl")
    base_row = joblib.load(HERE / "employee_attrition_base_row.pkl")
    return model, features, base_row

model, features, base_row = load_model_files()

st.markdown("""
<style>
.stApp {background-color: #f4f7fb;}
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {
    background: linear-gradient(120deg, #10243e, #155e75);
    padding: 30px; border-radius: 18px; color: white;
    margin-bottom: 24px;
}
.hero h1 {color: white;}
.hero p {color: #dbeafe;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <p> MACHINE LEARNING PROJECT</p>
    <h1>Employee Attrition Analytics</h1>
    <p>Random Forest • Predictive Analytics • HR Insights</p>
</div>
""", unsafe_allow_html=True)

st.write("Explore a sample employee profile and see the model's prediction.")

with st.form("employee_form"):
    st.subheader("Employee Profile")

    left, right = st.columns(2)

    with left:
        age = st.slider("Age", 18, 60, 30)
        distance = st.slider("Distance from home", 1, 29, 5)
        income = st.number_input(
            "Monthly income", min_value=100, max_value=100000,
            value=5000, step=500
        )

    with right:
        years = st.slider("Years at company", 0, 40, 3)
        satisfaction = st.select_slider(
            "Job satisfaction",
            options=[1, 2, 3, 4],
            value=3,
            help="1 = Low, 4 = Very High"
        )
        overtime = st.radio("Works overtime?", ["No", "Yes"])

    submitted = st.form_submit_button(
        "Analyze Employee Profile",
        use_container_width=True,
        type="primary"
    )

if submitted:
    employee = base_row.copy()

    employee["Age"] = age
    employee["MonthlyIncome"] = income
    employee["YearsAtCompany"] = years
    employee["JobSatisfaction"] = satisfaction
    employee["DistanceFromHome"] = distance

    if "OverTime_Yes" in employee.index:
        employee["OverTime_Yes"] = int(overtime == "Yes")

    employee_df = pd.DataFrame(
        [[employee[col] for col in features]],
        columns=features
    )

    probability = float(model.predict_proba(employee_df)[0][1])
    risk = "Higher" if probability >= 0.2 else "Lower"

    st.subheader("Prediction Results")
    result_col, probability_col = st.columns(2)

    with result_col:
        if risk == "Higher":
            st.warning("Higher predicted attrition risk")
        else:
            st.success("Lower predicted attrition risk")

    with probability_col:
        st.metric("Estimated attrition probability", f"{probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption("Demonstration classification threshold: 20%")

st.divider()
st.subheader("About this project")
st.write(
    "This student project uses a Random Forest classifier trained "
    "on the IBM HR Analytics dataset."
)

st.info(
    "Educational demonstration only. Other employee features are "
    "held at one sample employee's values. The probability is not "
    "independently calibrated or validated and must not be used to "
    "make real employment decisions."
)
