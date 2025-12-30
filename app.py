import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Config page
st.set_page_config(page_title="Chatbot RAG", layout="centered")

st.title("Chatbot RAG")
st.markdown("Responde perguntas baseadas nos PDFs carregados usando RAG.")

# Sidebar
with st.sidebar:
    st.header("Setup")
    
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.subheader("Arquivos")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    
    processar = st.button("Processar")

# Utils

def get_pdf_text(pdf_docs):
    # Extrai texto dos PDFs
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Splitter para chunks de 1000 chars
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key):
    # Gera embeddings e armazena no FAISS
    if not api_key:
        st.error("API Key necessaria")
        return None
        
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    # Setup da chain com memoria
    llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Main

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if processar and pdf_docs:
    if not api_key:
        st.warning("Insira a API Key")
    else:
        with st.spinner("Processando..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks, api_key)
            
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                st.success("Pronto")

# Chat Interface
if st.session_state.conversation:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Pergunta:")
    
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Gerando resposta..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            bot_response = response['answer']
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)

elif not pdf_docs:
    st.info("Carregue um PDF para iniciar")
