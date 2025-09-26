\from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's question clearly and directly."),
    ("user", "{question}")
])

# Streamlit UI
st.set_page_config(page_title="🤖 Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Hugging Face LLM
    llm = HuggingFaceTextGenInference(
        repo_id="tiiuae/falcon-7b-instruct",
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )

    try:
        # Generate response
        response = llm.invoke({"question": question})
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")
