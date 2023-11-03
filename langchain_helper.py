import os
#import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
load_dotenv()

llm = GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'],temperature=0)

#intiatiate embedding
instruct_embedding = HuggingFaceInstructEmbeddings()

vector_db_filepath = 'faiss_index'

def create_vector_db():
    #load csv file
    loder  = CSVLoader(file_path='data.csv',source_column='prompt', encoding = "ISO-8859-1")
    docs = loder.load()
    #create vectore store
    vectordb = FAISS.from_documents(documents=docs,embedding=instruct_embedding)
    vectordb.save_local(vector_db_filepath)

def get_qa_chain():
    #load vector store from local
    vectordb = FAISS.load_local(vector_db_filepath,instruct_embedding)

    #create the retriever the from databse
    retriever = vectordb.as_retriever()

    #intiate the prompt template
    template = """Given following context and a question, generate an answer based on that context only. In the answer try to provide as much as possible from that "response" section in the source document context  without  making. If the answer is  not found in the context , kindly state "I do not know ". Do  not  try to make up an answer
    CONTEXT: {context}
    QUESTION: {question}"""

    prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])
    
    #retrieve qa
    chain = RetrievalQA.from_chain_type(
                llm = llm,
                chain_type = 'stuff',
                retriever=retriever,
                input_key='query',
                return_source_documents=True,
                chain_type_kwargs={"prompt" :prompt}
                )
    return chain




if __name__ == '__main__':
    #create_vector_db()
    chain = get_qa_chain()
    print(chain('do you provide intership? Do you have EMI option'))


  