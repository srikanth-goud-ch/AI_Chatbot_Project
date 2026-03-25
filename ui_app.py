import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# UPDATED: These now come from langchain_classic
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Page Configuration
st.set_page_config(page_title="AI Doc Assistant", layout="wide")
st.title("📄 AI Documentation Assistant with Memory")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores the actual message objects
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 2. Sidebar for File Upload
with st.sidebar:
    st.header("Upload Section")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file and st.session_state.vectorstore is None:
        with st.spinner("Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.success("PDF Ready!")

# 3. RAG Setup (Only if vectorstore exists)
if st.session_state.vectorstore:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
    
    # Contextualize question: This rephrases the user's question to be standalone
    contextualize_q_system_prompt = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question."
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.vectorstore.as_retriever(), contextualize_q_prompt)

    # Answer question prompt
    qa_system_prompt = "You are an assistant for question-answering tasks. Use the retrieved context to answer. \n\n {context}"
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 4. Display Chat History
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # 5. Chat Input logic
    if user_query := st.chat_input("Ask about your document..."):
        # Display user message
        st.chat_message("user").markdown(user_query)
        
        # Generate response
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(answer)
            
        # Update history
        st.session_state.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=answer)
        ])
else:
    st.info("👈 Please upload a PDF in the sidebar to start chatting.")