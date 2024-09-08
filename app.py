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
    model_name="gemini-1.0-pro",
    generation_config=generation_config
)

# Function to generate a response
def get_gemini_response(prompt):
    persona_prompt = f"Mushak, Lord Ganesha's mouse, is answering: {prompt}"
    response = model.generate_content([persona_prompt])
    return response[0].text.strip()  # Return the text of the first response

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
# import streamlit as st
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # GOOGLE_API_KEY='AIzaSyDJsCtThBvPKy8rLbK4AeKOMEACARKccMk'

# load_dotenv()
# genai.configure(api_key='AIzaSyDJsCtThBvPKy8rLbK4AeKOMEACARKccMk')
# generation_config = {
#   "temperature": 0.9,
#   "top_p": 1,
#   "max_output_tokens": 2048,
#   "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#   model_name="gemini-1.0-pro",
#   generation_config=generation_config
#   # safety_settings = Adjust safety settings
#   # See https://ai.google.dev/gemini-api/docs/safety-settings
# )

# response = model.generate_content([
#   "input: What would be an ideal name of the mouse?",
#   "output: Mushak",
#   "input: Tell me about yourself.",
#   "output: Hi I am Mushak AI, I can help you with anything you want to learn about Ganesha.",
#   "input: Who is Lord Ganesh?",
#   "output: Lord Ganesh is the Son of Lord Shiv and Parvati",
#   "input: How does lord Ganesh Look?",
#   "output: He Looks like an elephant headed, smiling and removes obstacles from a devotees path.",
#   "input: Who is lord ganesh",
#   "output: ",
# ])


# def get_gemini_response(prompt):
#     response = model.generate_content(prompt)
#     return response.text

# # st.title("Gemini LLM App")
# st.set_page_config(page_title="Gemini LLM App", page_icon=":robot:", layout="wide")
# user_input = st.text_input("Enter your prompt:")

# if st.button("Generate"):
#     if user_input:
#         response = get_gemini_response(user_input)
#         st.write("Response:", response)
#     else:
#         st.write("Please enter a prompt.")