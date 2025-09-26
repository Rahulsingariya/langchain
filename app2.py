from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's question clearly and directly."),
    ("user", "{question}")
])

st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask me anything..."):
   
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    
    llm = Ollama(model="llama2")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        response = chain.invoke({"question": question})

        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")