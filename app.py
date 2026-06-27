import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("burnout_prediction_model.pkl")

st.title("Student Burnout Risk Prediction")

# Input fields

Major_Category = st.selectbox(
    "Major Category",
    ["STEM", "Business", "Medical", "Arts", "Humanities"]
)

Year_of_Study = st.selectbox(
    "Year of Study",
    ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
)

Pre_Semester_GPA = st.number_input(
    "Pre Semester GPA",
    min_value=0.0,
    max_value=4.0,
    value=3.0
)

Weekly_GenAI_Hours = st.number_input(
    "Weekly GenAI Hours",
    min_value=0.0,
    value=5.0
)

Primary_Use_Case = st.selectbox(
    "Primary Use Case",
    [
        "Copywriting/Drafting",
        "Debugging/Troubleshooting",
        "Direct_Answer_Generation",
        "Summarizing_Reading",
        "Ideation"
    ]
)

Prompt_Engineering_Skill = st.selectbox(
    "Prompt Engineering Skill",
    ["Beginner", "Intermediate", "Advanced"]
)

Tool_Diversity = st.slider(
    "Tool Diversity",
    1,5,3
)

Paid_Subscription = st.selectbox(
    "Paid Subscription",
    [True, False]
)

Traditional_Study_Hours = st.number_input(
    "Traditional Study Hours",
    min_value=0.0,
    value=10.0
)

Perceived_AI_Dependency = st.slider(
    "Perceived AI Dependency",
    1,10,5
)

Institutional_Policy = st.selectbox(
    "Institutional Policy",
    [
        "Allowed_With_Citation",
        "Actively_Encouraged",
        "Strict_Ban"
    ]
)

Anxiety_Level_During_Exams = st.slider(
    "Anxiety Level During Exams",
    1,10,5
)

Post_Semester_GPA = st.number_input(
    "Post Semester GPA",
    min_value=0.0,
    max_value=4.0,
    value=3.0
)

Skill_Retention_Score = st.number_input(
    "Skill Retention Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

if st.button("Predict Burnout Risk"):

    input_df = pd.DataFrame({
        'Major_Category':[Major_Category],
        'Year_of_Study':[Year_of_Study],
        'Pre_Semester_GPA':[Pre_Semester_GPA],
        'Weekly_GenAI_Hours':[Weekly_GenAI_Hours],
        'Primary_Use_Case':[Primary_Use_Case],
        'Prompt_Engineering_Skill':[Prompt_Engineering_Skill],
        'Tool_Diversity':[Tool_Diversity],
        'Paid_Subscription':[Paid_Subscription],
        'Traditional_Study_Hours':[Traditional_Study_Hours],
        'Perceived_AI_Dependency':[Perceived_AI_Dependency],
        'Institutional_Policy':[Institutional_Policy],
        'Anxiety_Level_During_Exams':[Anxiety_Level_During_Exams],
        'Post_Semester_GPA':[Post_Semester_GPA],
        'Skill_Retention_Score':[Skill_Retention_Score]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Burnout Risk: {prediction}")