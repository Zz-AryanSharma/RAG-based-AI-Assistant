import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="IEEE RAS Assistant", page_icon="🤖")

# --- INITIALIZATION ---
# Fallback if API key is not yet set
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state.GEMINI_API_KEY = ""

with st.sidebar:
    st.title("Settings")
    # Allow user to input key if not running with secrets configured yet
    api_key = st.text_input("Gemini API Key", type="password", help="Get your API key from Google AI Studio")
    if api_key:
        st.session_state.GEMINI_API_KEY = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GEMINI_API_KEY" in st.secrets: 
        st.session_state.GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

    st.markdown("---")
    st.markdown("Built for **IEEE RAS** using publicly available information.")

# Halt if no API key is available
if not st.session_state.GEMINI_API_KEY:
    st.warning("Please enter your Gemini API Key in the sidebar to continue.")
    st.stop()


@st.cache_resource
def get_vectorstore():
    """Loads documents, or loads the existing vector store if it exists."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    persist_directory = "./chroma_db"
    
    # Check if the database already exists on disk
    if os.path.exists(persist_directory):
        # Load the existing vector store
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
        
    # Otherwise, load documents and create it
    data_dir = "." 
    
    # Load all .txt files
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()

    # Split Text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create Embeddings and Vector Store, persisting it to disk
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore

def get_rag_chain():
    """Sets up the retrieval chain with LLM and prompt."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    system_prompt = (
        "You are a helpful AI assistant for the IEEE Robotics and Automation Society (RAS). "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, just say that you don't know. "
        "Keep the answer concise and professional.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- UI MAIN ---
st.title("IEEE RAS AI Assistant 🤖")
st.write("Ask me anything about IEEE RAS (Membership, Conferences, Technical Committees, etc.) based on the provided documents.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What are the benefits of membership?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    rag_chain = get_rag_chain()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")

