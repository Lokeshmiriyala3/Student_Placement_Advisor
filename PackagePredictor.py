import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import base64
import json  # To parse the response into a dictionary

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Path to the local image (you can also upload an image in Streamlit directly)
image_path = "app2b1.jpg"  # Update this to the image you want to use

# Convert image to base64
image_base64 = image_to_base64(image_path)

# CSS with the embedded Base64 image
background_css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    position: relative; /* Needed for the overlay */
    background-image: url("data:image/jpg;base64,{image_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    color: white; /* Ensures text is readable */
    height: 100vh; /* Ensure the container takes the full viewport height */
    overflow: auto; /* Allows content to scroll if needed */
}}

/* Adding the transparent overlay */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5); /* Adjust 0.5 for transparency level */
    z-index: 1; /* Places the overlay above the image */
}}

[data-testid="stAppViewContainer"] > * 
{{
    position: relative;
    z-index: 2; /* Ensures content stays on top of the overlay */
    padding: 20px; /* Adjusts padding to ensure content is not hidden behind the overlay */
}}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(background_css, unsafe_allow_html=True)
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

input_prompt = """
Hey, act like a skilled and experienced ATS (Applicant Tracking System) with a deep understanding of the tech field, software engineering, data science, data analytics, and big data engineering. Your task is to evaluate the resume based on the given job description.

Consider that the job market is very competitive, so you should provide the best assistance for improving the resumes. Assign a percentage match based on the job description and identify any missing keywords with high accuracy.

resume: {text}
description: {jd}

I want the response in one single string, structured as follows:
{{"JD Match": "%", "MissingKeywords": [], "Profile Summary": ""}}
"""

st.title("Smart ATS")
st.text("Improve Your Resume for ATS")
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        response = get_gemini_response(input_prompt)
        
        # Print the raw response to inspect its structure
        st.write("Raw Response:")
        st.code(response)
        
        # Clean the response by removing extra curly braces
        clean_response = response.strip().replace("{{", "{").replace("}}", "}")

        # Try to parse the cleaned response and display it neatly
        try:
            # Convert the cleaned response string to a dictionary
            response_dict = json.loads(clean_response)

            # Display the results in a neat way
            st.subheader("Job Description Match Result")
            st.markdown(f"**JD Match**: {response_dict.get('JD Match', 'N/A')}")
            st.markdown(f"**Missing Keywords**: {', '.join(response_dict.get('MissingKeywords', [])) if response_dict.get('MissingKeywords') else 'None'}")
            st.markdown(f"**Profile Summary**: {response_dict.get('Profile Summary', 'N/A')}")
        except json.JSONDecodeError:
            st.write("Error parsing the response. Please check the format of the generated response.")
