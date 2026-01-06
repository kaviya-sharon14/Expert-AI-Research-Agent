import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import datetime
import os

# --- UI Setup ---
st.set_page_config(page_title="Expert AI Research Agent", layout="wide")
st.title("🤖 Multi-Tool Research Agent")
st.markdown("### Advanced RAG + Automated Data Logging")

# --- 1. Knowledge Base Logic (Folder + Uploads) ---
@st.cache_resource
def process_knowledge_base(uploaded_files):
    all_docs = []
    
    # A. AUTOMATED LOADING: Load from 'data' folder if files exist
    DATA_PATH = "data/"
    if os.path.exists(DATA_PATH) and any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        folder_loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
        all_docs.extend(folder_loader.load())

    # B. MANUAL UPLOAD: Load files from the Sidebar
    if uploaded_files:
        for file in uploaded_files:
            with open("temp.pdf", "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            all_docs.extend(loader.load())
    
    if not all_docs:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- UI Sidebar ---
with st.sidebar:
    st.header("📁 Knowledge Base")
    files = st.file_uploader("Upload Additional PDFs", accept_multiple_files=True)
    
    # Process both folder and uploaded files
    with st.spinner("Indexing system knowledge..."):
        db = process_knowledge_base(files)
    
    if db:
        st.success("Knowledge Base Ready!")
    else:
        st.warning("Please add PDFs to the 'data' folder or upload here.")

# --- 2. The Agent Logic ---
query = st.text_input("Ask the Agent to analyze something:")

if query:
    if db:
        results = db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in results])
        
        with st.expander("👁️ View Agent Reasoning"):
            st.write("1. Searching local vector database (Folder + Uploads)...")
            st.write(f"2. Found {len(results)} relevant segments.")
            st.write("3. Synthesizing data for logging...")

        st.subheader("Final Analysis")
        st.info(f"**Context found in Documents:**\n\n{context[:1000]}...")
        
        # --- 3. The "Action" (Automated Logging) ---
        st.divider()
        if st.button("Confirm & Log to System"):
            log_filename = "audit_log.csv"
            # Logic to handle source name
            source_name = "System Knowledge" if not files else files[0].name
            
            log_entry = {
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Query": query,
                "Verdict": "Analysis Verified",
                "Source": source_name
            }
            df = pd.DataFrame([log_entry])
            file_exists = os.path.isfile(log_filename)
            df.to_csv(log_filename, mode='a', index=False, header=not file_exists)
            st.success(f"✅ Entry added to `{log_filename}` successfully!")
    else:
        st.error("Knowledge base is empty. Please provide documents.")
