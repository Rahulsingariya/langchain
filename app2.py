from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# System + user prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's question clearly and directly."),
    ("user", "{question}")
])

# Streamlit app
st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot")

# Initialize session state for chat display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages (UI only, not used in LLM call)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask me anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Initialize Ollama Llama2 model
    llm = Ollama(model="llama2")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    try:
        # Generate independent answer
        response = chain.invoke({"question": question})

        # Show assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")
