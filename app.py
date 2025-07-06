import streamlit as st
import os
import tempfile
import numpy as np
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_DIM = 384

@st.cache_resource
def load_models():
    """Load and cache the embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL)

def extract_pdf_content(pdf_path):
    """Extract text and tables from PDF"""
    content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                content.append(f"Page {page_num+1} Text:\n{text}")
            
            # Extract tables
            tables = page.extract_tables()
            for table_num, table in enumerate(tables):
                if table:
                    # Convert table to readable format
                    table_text = f"Page {page_num+1} Table {table_num+1}:\n"
                    for i, row in enumerate(table):
                        clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                        table_text += "| " + " | ".join(clean_row) + " |\n"
                        if i == 0:  # Add separator after header
                            table_text += "| " + " | ".join(["---"] * len(row)) + " |\n"
                    content.append(table_text)
    
    return content

def create_chunks(content, chunk_size=500, overlap=50):
    """Split content into overlapping chunks"""
    chunks = []
    for item in content:
        words = item.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
    return chunks

def create_vector_store(pdf_path, embedding_model):
    """Process PDF and create vector store"""
    # Extract content
    content = extract_pdf_content(pdf_path)
    
    # Create chunks
    chunks = create_chunks(content)
    
    # Generate embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, chunks

def search_similar(query, index, chunks, embedding_model, k=5):
    """Find similar chunks using vector search"""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_answer(query, context, groq_api_key):
    """Generate answer using Groq API"""
    client = Groq(api_key=groq_api_key)
    
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
    If the answer involves tables or data, present it clearly and accurately.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="PDF RAG Assistant",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF RAG Assistant")
    st.markdown("Upload a PDF and ask questions about its content!")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key"
        )
        
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key")
            st.stop()
    
    # Load embedding model
    embedding_model = load_models()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with col2:
        st.header("❓ Ask Question")
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What is the main topic?\nWhat data is in the tables?\nSummarize the key points..."
        )
    
    # Process and answer
    if uploaded_file and question:
        if st.button("🔍 Get Answer", type="primary"):
            with st.spinner("Processing PDF and generating answer..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name
                    
                    # Create vector store
                    index, chunks = create_vector_store(pdf_path, embedding_model)
                    
                    # Search for relevant content
                    relevant_chunks = search_similar(question, index, chunks, embedding_model)
                    context = "\n\n".join(relevant_chunks)
                    
                    # Generate answer
                    answer = generate_answer(question, context, groq_api_key)
                    
                    # Display results
                    st.header("📝 Answer")
                    st.write(answer)
                    
                    # Show relevant context (optional)
                    with st.expander("🔍 View Retrieved Context"):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.text_area(f"Context {i}", chunk, height=100)
                    
                    # Clean up
                    os.unlink(pdf_path)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Examples section
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **For Text Content:**
        - What is the main topic of this document?
        - Summarize the key findings
        - What are the conclusions?
        
        **For Table Content:**
        - What is the total revenue shown in the table?
        - List all the values in the first column
        - What data is presented in the tables?
        
        **General Questions:**
        - Give me an overview of the document
        - What are the most important points?
        """)
    
    # Instructions
    with st.expander("📋 How to Use"):
        st.markdown("""
        1. **Enter your Groq API Key** in the sidebar
        2. **Upload a PDF file** using the file uploader
        3. **Type your question** in the text area
        4. **Click "Get Answer"** to process and get results
        
        **Features:**
        - ✅ Extracts text and tables from PDFs
        - ✅ Uses all-MiniLM-L6-v2 for embeddings
        - ✅ Uses Llama-4-Scout via Groq API
        - ✅ Fast FAISS vector search
        - ✅ Handles both text and tabular questions
        """)

if __name__ == "__main__":
    main()