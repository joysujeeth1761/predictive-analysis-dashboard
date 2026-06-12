import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Load Model Artifacts
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "student_performance_model.pkl"
)

scaler = joblib.load(
    BASE_DIR / "models" / "scaler.pkl"
)

feature_names = joblib.load(
    BASE_DIR / "models" / "feature_names.pkl"
)

# ----------------------------
# UI
# ----------------------------

st.title("🎓 Student Academic Performance Predictor")

st.write(
    "Predict final student grades using Machine Learning"
)

# Numerical Features

age = st.slider(
    "Age",
    15, 22, 17,
    help="Student's age in years."
)

Medu = st.slider(
    "Mother Education",
    0, 4, 2,
    help="""
    Mother's education level:
    0 = None
    1 = Primary Education
    2 = 5th–9th Grade
    3 = Secondary Education
    4 = Higher Education
    """
)

Fedu = st.slider(
    "Father Education",
    0, 4, 2,
    help="""
    Father's education level:
    0 = None
    1 = Primary Education
    2 = 5th–9th Grade
    3 = Secondary Education
    4 = Higher Education
    """
)

traveltime = st.slider(
    "Travel Time",
    1, 4, 2,
    help="""
    Time taken to travel from home to school:
    1 = <15 min
    2 = 15–30 min
    3 = 30–60 min
    4 = >60 min
    """
)

studytime = st.slider(
    "Study Time",
    1, 4, 2,
    help="""
    Weekly study time:
    1 = <2 hours
    2 = 2–5 hours
    3 = 5–10 hours
    4 = >10 hours
    """
)

failures = st.slider(
    "Past Failures",
    0, 4, 0,
    help="Number of past class failures."
)

famrel = st.slider(
    "Family Relationship",
    1, 5, 3,
    help="""
    Quality of family relationships:
    1 = Very Bad
    2 = Bad
    3 = Average
    4 = Good
    5 = Excellent
    """
)

freetime = st.slider(
    "Free Time",
    1, 5, 3,
    help="""
    Free time after school:
    1 = Very Low
    5 = Very High
    """
)

goout = st.slider(
    "Going Out Frequency",
    1, 5, 3,
    help="""
    Frequency of going out with friends:
    1 = Very Rarely
    5 = Very Frequently
    """
)

Dalc = st.slider(
    "Weekday Alcohol Consumption",
    1, 5, 1,
    help="""
    Alcohol consumption on weekdays:
    1 = Very Low
    5 = Very High
    """
)

Walc = st.slider(
    "Weekend Alcohol Consumption",
    1, 5, 1,
    help="""
    Alcohol consumption on weekends:
    1 = Very Low
    5 = Very High
    """
)

health = st.slider(
    "Health Status",
    1, 5, 3,
    help="""
    Current health condition:
    1 = Very Poor
    5 = Excellent
    """
)

absences = st.number_input(
    "Absences",
    min_value=0,
    max_value=100,
    value=0,
    help="Total number of school absences."
)


# categorical values


st.header("📝 Student Profile")

sex = st.selectbox(
"Sex",
["F", "M"],
help="Student's gender."
)

sex_labels = {
"F": "Female",
"M": "Male"
}

st.caption(f"Selected: {sex_labels[sex]}")

school = st.selectbox(
"School",
["GP", "MS"],
help="School attended by the student."
)

school_labels = {
"GP": "Gabriel Pereira School",
"MS": "Mousinho da Silveira School"
}

st.caption(f"Selected: {school_labels[school]}")

address = st.selectbox(
"Address Type",
["R", "U"],
help="Type of residential area."
)

address_labels = {
"R": "Rural Area",
"U": "Urban Area"
}

st.caption(f"Selected: {address_labels[address]}")

higher = st.selectbox(
"Higher Education Intention",
["no", "yes"],
help="Does the student want to pursue higher education?"
)

higher_labels = {
"yes": "Plans for Higher Education",
"no": "No Plans for Higher Education"
}

st.caption(f"Selected: {higher_labels[higher]}")

internet = st.selectbox(
"Internet Access at Home",
["no", "yes"],
help="Whether the student has internet access at home."
)

internet_labels = {
"yes": "Internet Available",
"no": "No Internet Access"
}

st.caption(f"Selected: {internet_labels[internet]}")

romantic = st.selectbox(
"Romantic Relationship",
["no", "yes"],
help="Whether the student is currently in a romantic relationship."
)

romantic_labels = {
"yes": "Currently in a Relationship",
"no": "Not in a Relationship"
}

st.caption(f"Selected: {romantic_labels[romantic]}")


# ----------------------------
# Predict Button
# ----------------------------

if st.button("Predict Grade"):

    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    # Numerical

    input_df["age"] = age
    input_df["Medu"] = Medu
    input_df["Fedu"] = Fedu
    input_df["traveltime"] = traveltime
    input_df["studytime"] = studytime
    input_df["failures"] = failures
    input_df["famrel"] = famrel
    input_df["freetime"] = freetime
    input_df["goout"] = goout
    input_df["Dalc"] = Dalc
    input_df["Walc"] = Walc
    input_df["health"] = health
    input_df["absences"] = absences

    # Engineered Features

    input_df["parent_edu"] = Medu + Fedu

    input_df["total_alcohol"] = Dalc + Walc

    input_df["study_failure_ratio"] = (
        studytime / (failures + 1)
    )

    # Encoded Features

    if school == "MS":
        input_df["school_MS"] = 1

    if sex == "M":
        input_df["sex_M"] = 1

    if address == "U":
        input_df["address_U"] = 1

    if higher == "yes":
        input_df["higher_yes"] = 1

    if internet == "yes":
        input_df["internet_yes"] = 1

    if romantic == "yes":
        input_df["romantic_yes"] = 1

    # Scale

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(
        scaled_input
    )[0]

    st.success(
        f"Predicted Final Grade: {prediction:.2f} / 20"
    )