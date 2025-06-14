import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
#from config import CHAT_PROMPT_TEMPLATE
import json

# Title on the page
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50; font-family: Arial;'>Sravanth🪶</h2>",
    unsafe_allow_html=True,
)

CHAT_PROMPT_TEMPLATE = """
You are Sravanth from Accenture. 
You know everything and aim to provide a concise response.
User: {question}
Assistant: """

template = CHAT_PROMPT_TEMPLATE
prompt = ChatPromptTemplate.from_template(template)

#Load the local Llama3.2 model that we pulled using ollama
model = OllamaLLM(model="llama3.2",base_url="https://blank-app-hh8c673f4v9.streamlit.app/")
chain = prompt | model

#Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How may I help you?"}
    ]

#Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Handle user input
if user_input := st.chat_input("What is up?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    #with st.chat_message("assistant"):
    #    response = chain.invoke({"question": user_input})
    #    st.markdown(response)
    try:
        response = chain.invoke({"question": user_input})
        if isinstance(response, dict) and "text" in response:
            assistant_response = response["text"]
        else:
            assistant_response = "Unexpected response format."
    except json.JSONDecodeError as e:
        assistant_response = "Error decoding JSON response."
        st.write(f"JSONDecodeError: {e}")
    except Exception as e:
        assistant_response = "An error occurred while processing your request."
        st.write(f"Error: {e}")

    st.markdown(assistant_response)


    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
