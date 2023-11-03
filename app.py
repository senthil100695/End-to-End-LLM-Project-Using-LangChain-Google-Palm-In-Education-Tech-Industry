import streamlit as st
from langchain_helper import create_vector_db,get_qa_chain

st.title('Customer QA Bot')
#st.set_page_config(page_title="Customer QA Bot 😃")
btn = st.button('Create Knowledge Base')


if btn:
    pass

question = st.text_input('Question: 😃')

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header('Answer: ')

    st.write(response['result'])
