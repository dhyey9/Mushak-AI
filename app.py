import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))  # Use environment variable for safety

generation_config = {
    "temperature": 0.7,  # Adjust temperature to balance creativity and relevance
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

# Define the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Function to generate a response
def get_gemini_response(prompt):
    response = model.generate_content([prompt])
    return response.text.strip()  # Directly access the text attribute

# Streamlit app interface
st.set_page_config(page_title="Gemini LLM App", page_icon=":robot:", layout="wide")
st.title("Mushak AI - Your Guide to Lord Ganesha")

user_input = st.text_input("Ask Mushak a question about Lord Ganesha:")

if st.button("Generate"):
    if user_input:
        response = get_gemini_response(user_input)
        st.write("Mushak's Response:", response)
    else:
        st.write("Please enter a prompt.")
