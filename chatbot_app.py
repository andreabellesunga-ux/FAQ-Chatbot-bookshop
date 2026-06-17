import streamlit as st
from retriever import find_best_match

st.set_page_config(page_title="Riverside Books Chatbot", layout="centered")

st.title("Riverside Books Chatbot")
st.markdown("Ask me anything about the bookshop!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    answer, score = find_best_match(prompt)

    with st.chat_message("assistant"):
        if answer:
            st.markdown(answer)
        else:
            st.markdown("Sorry, I don’t know that one - please ask a member of staff.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer if answer else "Sorry..."
    })

with st.sidebar:
    st.header("About")
    st.write("AI chatbot using embedding-based semantic search (OpenAI).")