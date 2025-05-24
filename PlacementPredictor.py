from tkinter import font
import streamlit as st
import numpy as np
import pandas as pd
import base64
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the trained model and scaler
model = joblib.load("ModelRF.pkl")
scaler = joblib.load("scaler.pkl")

# Helper function for image-to-base64 conversion
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    image_base64 = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{image_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            color: white;
            height: 100vh;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5); /* Adjust 0.5 for transparency level */
            z-index: 1;
        }}
        [data-testid="stAppViewContainer"] > * {{
            position: relative;
            z-index: 2;
            padding: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Initial background
background_image_path = "app1b2.jpg"
success_image_path = "success.jpg"
failure_image_path = "f1.jpg"

# Session state to manage app flow
if "result_displayed" not in st.session_state:
    st.session_state.result_displayed = False
    st.session_state.placement_status = None

if st.session_state.result_displayed:
    if st.session_state.placement_status == "Placed":
        set_background(success_image_path)
        st.markdown("<h1 style='text-align: center;'>🎉 Congratulations! 🎉</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; font-size: 20px;'>You are likely to be placed. Best of luck for your future endeavors!</p>",
            unsafe_allow_html=True,
        )
    else:
        set_background(failure_image_path)
        st.markdown("<h1 style='text-align: center;'>❌ Not Placed</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; font-size: 20px;'>Don't be discouraged! Keep working hard and success will come.</p>",
            unsafe_allow_html=True,
        )

else:
    # Main page logic
    
    
    # Main page logic
    set_background(background_image_path)
    st.title("Student Placement Prediction")
    name = st.text_input("Full Name")
    year_of_graduation = st.selectbox("Year Of Graduation", ["2025", "2024", "2023"])
    department = st.selectbox("Department", ["CSE", "IT", "ECE", "CIVIL", "EEE", "MECH", "AIDS", "CSBS", "CSD", "AIML", "Other"])
    mobile_number = st.text_input("Mobile Number")
    percentage_10th = st.number_input("10th Percentage", min_value=0.0, max_value=100.0)
    percentage_12th = st.number_input("12th Percentage", min_value=0.0, max_value=100.0)
    seat_allocation = st.selectbox("BTech Seat Allocation", ["Convenor quota (EAMCET)", "Management quota"])
    cgpa_btech = st.number_input("CGPA in B.Tech", min_value=0.0, max_value=10.0)
    eamcet_rank = st.number_input("EAMCET Rank", min_value=0, max_value=200000)
    certifications_completed = st.selectbox("How many certification courses completed during BTech (Other than NPTEL)?", [str(i) for i in range(11)] + ["More than 10"])
    nptel_courses = st.selectbox("How many NPTEL courses completed during BTech?", ["0", "1", "2", "3", "4", "5", "More than 5"])
    gaming = st.selectbox("Did you engage in online gaming during BTech?", ["Yes", "No"])
    hours_gaming = st.selectbox("Hours spent daily on online gaming", ["0", "1", "2", "3", "4", "More than 5"])
    physical_activities = st.selectbox("Did you do physical activities like sports during BTech?", ["Yes", "No"])
    hours_physical = st.selectbox("Hours spent daily on physical activities", ["0", "1", "2", "3", "More than 3"])
    coding_skills = st.slider("How would you rate your knowledge and skills in coding and programming on the scale of 5?", min_value=0, max_value=5)
    hours_coding = st.selectbox("How many hours per day did you dedicate to problem-solving or coding during your BTech program on an average ?", ["0", "1", "2", "3", "4", "More than 5"])
    projects_completed = st.selectbox("How many projects have you completed during your BTech program on your own?", ["0", "1", "2", "3", "4", "5", "More than 5"])
    hackathons_participated = st.selectbox("How many hackathons did you participate in during your BTech program?", ["0", "1", "2", "3", "4", "5", "More than 5"])
    hackathons_won = st.selectbox("How many hackathons did you win during your BTech program ?(any prize)", ["0", "1", "2", "3", "4", "5", "More than 5"])
    hours_placement_preparation = st.selectbox("How many hours per day did you spend throughout your BTech program preparing for campus placements, including aptitude, reasoning, verbal skills, coding, technical on an average?", ["0", "1", "2", "3", "4", "5", "More than 5"])
    mock_interviews = st.selectbox("How many mock interviews did you attend for placements during your BTech program?", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "More than 10"])
    communication_skills = st.slider("How would you rate your communication skills on the scale of 5?", min_value=1, max_value=5)
    backlogs = st.selectbox("How many backlogs did you have throughout your entire BTech program?", ["0", "1", "2", "3", "4", "5", "More than 5"])
    cleared_backlogs_years = st.selectbox("In how many years after 4 year BTech program did you clear your backlogs?", ["0", "1", "2", "3", "4", "More than 5"])
    merit_scholarship = st.selectbox("Did you receive any merit scholarship during BTech program apart from regular  government fee reimbursement?", ["Yes", "No"])
    hours_social_media = st.selectbox("How many hours per day did you spend on social media platforms during your BTech program on an average?", [str(i) for i in range(11)])
    interview_confidence = st.slider("How would you rate your confidence levels during interviews on the scale of 5 ?", min_value=0, max_value=5)
    stipend_received = st.selectbox("Did you receive any stipend from the companies during your final year internship ?", ["Yes", "No"])
    tech_skills = st.slider("How would you rate yourself in a particular technology (e.g., full stack, front-end, back-end, cloud, AI/ML, app development)?", min_value=0, max_value=5)
    coding_competitions = st.selectbox("How many coding competition did you participate during your BTech program ?", ["0", "1", "2", "3", "4", "5"])
    project_contribution = st.selectbox("What is the percentage of contribution in your final year project ?", ["0-20", "20-40", "40-60", "60-80", "80-100"])
    real_time_project = st.selectbox("Are you confident that your final year project problem statement taken is a real-time project and not copied from any online resources?", ["Yes", "No"])
    family_income = st.selectbox("What is your annual family income ?", ["Less than 2 lakhs", "2-5 lakhs", "5-10 lakhs", "10-20 lakhs", "Greater than 20 lakhs"])
    attendance = st.selectbox("What is average Attendance percentage did you maintained in your B.Tech program", ["Less than 50", "50-65", "65-75", "75-85", "85-100"])
    siblings = st.selectbox("How many siblings do you have ?", ["0", "1", "2"])
    faculty_doubts = st.selectbox("Do you have habit of asking doubts to your faculty", ["Yes", "No"])
    area = st.selectbox("From which area are you from", ["Urban", "Rural"])
    medium = st.selectbox("In which medium did you studied upto 10th standard", ["English", "Telugu"])
    father_occupation = st.selectbox("What is your father's occupation ?", ["Farmer", "Business", "Government employee", "Private employee", "Labour"])
    classmate_doubts = st.selectbox("Do you have habit of asking doubts of your classmates or friends", ["Yes", "No", "Maybe"])
    public_speaking = st.slider("What were your confidence levels in public speaking on a scale of 5?", min_value=0, max_value=5)
    if st.button("Predict Placement Status"):
        data = [
        year_of_graduation,percentage_10th, percentage_12th, seat_allocation,cgpa_btech, eamcet_rank,  interview_confidence, 
        certifications_completed, nptel_courses, gaming, hours_gaming, physical_activities, hours_physical,coding_skills,
        hours_coding, projects_completed, hackathons_participated, hackathons_won, hours_placement_preparation, mock_interviews,communication_skills ,
        backlogs, cleared_backlogs_years, merit_scholarship, hours_social_media, stipend_received,
        tech_skills, coding_competitions, project_contribution, real_time_project, family_income, attendance, siblings,
        faculty_doubts, area, medium, father_occupation, classmate_doubts ,public_speaking]
        le = LabelEncoder()
        data = [le.fit_transform([x])[0] if isinstance(x, str) else x for x in data]

        # Scale the data
        scaled_data = scaler.transform([data])

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        st.session_state.result_displayed = True

        # Handle binary classification
        st.session_state.placement_status = "Placed" if prediction == 0 else "Not Placed"
        st.rerun()
        
    
        st.rerun()
    
        


    



    

# Inject the CSS into the Streamlit app
# Load the trained model and scaler
