import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from httpx import ConnectError, RequestError

# Title on the page
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50; font-family: Arial;'>Hermione 🪶</h2>",
    unsafe_allow_html=True,
)

# Define the prompt template
CHAT_PROMPT_TEMPLATE = """
You are Hermione Granger from Harry Potter. 
You know everything and aim to provide a concise response.
User: {question}
Assistant: """

template = CHAT_PROMPT_TEMPLATE
prompt = ChatPromptTemplate.from_template(template)

# Load the local Llama3.2 model using Ollama
try:
    model = OllamaLLM(model="llama3.2")
    chain = prompt | model
except Exception as e:
    st.error("Failed to initialize the model.")
    st.error(f"Error details: {str(e)}")
    st.stop()

# Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How may I help you?"}
    ]

# Display the chat history
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
    with st.chat_message("assistant"):
        try:
            response = chain.invoke({"question": user_input})
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except ConnectError as e:
            st.error("Connection to the Ollama server failed. Please check the server.")
            st.error(f"Details: {str(e)}")
        except RequestError as e:
            st.error("An HTTP request error occurred.")
            st.error(f"Details: {str(e)}")
        except Exception as e:
            st.error("An unexpected error occurred while generating the response.")
            st.error(f"Details: {str(e)}")
