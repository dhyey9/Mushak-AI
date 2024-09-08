import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

# Define the model
model = genai.GenerativeModel(
    model_name="tunedModels/rolespecificconversationslordganesha-7kh",
    generation_config=generation_config
)

# Function to generate a response
def get_gemini_response(prompt):
    response = model.generate_content([prompt])
    return response.text.strip()

# Function to save chat history
def save_chat_history(prompt, response):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((prompt, response))

# Streamlit app interface
st.set_page_config(page_title="Mushak AI", page_icon="🐀", layout="wide")
st.title("Mushak AI - Your Guide to Lord Ganesha")

# Sidebar for chat history
with st.sidebar:
    st.header("Chat History")
    if 'chat_history' in st.session_state:
        for i, (p, r) in enumerate(st.session_state.chat_history):
            if st.button(f"Chat {i+1}: {p[:30]}...", key=f"history_{i}"):
                st.session_state.selected_chat = i

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Ask Mushak a question about Lord Ganesha:", key="user_input")

    if st.button("Generate", key="generate"):
        if user_input:
            response = get_gemini_response(user_input)
            save_chat_history(user_input, response)
            st.write("Mushak's Response:", response)
        else:
            st.write("Please enter a prompt.")

    # Display selected chat history
    if 'selected_chat' in st.session_state:
        i = st.session_state.selected_chat
        st.write(f"Selected Chat: {st.session_state.chat_history[i][0]}")
        st.write(f"Response: {st.session_state.chat_history[i][1]}")

with col2:
    st.subheader("Suggested Prompts")
    prompts = [
        "Good morning, can you give me some information on Ganesha?",
        "How is Ganesha related to other Hindu gods?",
        "What are the different names of Lord Ganesha?",
        "Why is Ganesha worshipped before starting any new venture?",
        "Why does Ganesha have one broken tusk?",
        "Why is Ganesha called the remover of obstacles?",
        "What are the symbols associated with Lord Ganesha?"
    ]
    
    selected_prompt = st.selectbox("Select a prompt", [""] + prompts, key="prompt_select")
    if selected_prompt:
        st.write(f"Selected prompt: {selected_prompt}")
        if st.button("Use this prompt"):
            response = get_gemini_response(selected_prompt)
            save_chat_history(selected_prompt, response)
            st.write("Mushak's Response:", response)
# import streamlit as st
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Configure the API key
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))  # Use environment variable for safety

# generation_config = {
#     "temperature": 0.7,  # Adjust temperature to balance creativity and relevance
#     "top_p": 1,
#     "max_output_tokens": 2048,
#     "response_mime_type": "text/plain",
# }

# # Define the model
# model = genai.GenerativeModel(
#     model_name="tunedModels/rolespecificconversationslordganesha-7kh",
#     generation_config=generation_config
# )

# # Function to generate a response
# def get_gemini_response(prompt):
#     response = model.generate_content([prompt])
#     return response.text.strip()  # Directly access the text attribute

# # Streamlit app interface
# st.set_page_config(page_title="Gemini LLM App", page_icon=":robot:", layout="wide")
# st.title("Mushak AI - Your Guide to Lord Ganesha")

# user_input = st.text_input("Ask Mushak a question about Lord Ganesha:")

# if st.button("Generate"):
#     if user_input:
#         response = get_gemini_response(user_input)
#         st.write("Mushak's Response:", response)
#     else:
#         st.write("Please enter a prompt.")
