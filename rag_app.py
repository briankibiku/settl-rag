import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
load_dotenv()

# Verify API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set in .env file")

# Initialize FastAPI
app = FastAPI(title="RAG PDF Query API", version="1.0.0")

# Add CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                    # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],                    # Allow all HTTP methods
    allow_headers=["*"],                    # Allow all headers
)

# Global variables for vector store
vector_store = None
retriever = None
llm = None

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list

# ==================== RAG Functions ====================

def load_pdfs_from_folder(folder_path: str = "documents"):
    """Load all PDFs from a folder."""
    print(f"Loading PDFs from {folder_path}...")
    docs = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Creating {folder} folder...")
        folder.mkdir(exist_ok=True)
        print(f"Please place your PDF files in the '{folder}' folder")
        return docs
    
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return docs
    
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    
    print(f"Total pages loaded: {len(docs)}\n")
    return docs

def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into chunks."""
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks\n")
    return chunks

def create_vector_store(chunks: list):
    """Create and store embeddings in Chroma."""
    print("Creating embeddings and storing in Chroma...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf-documents",
        persist_directory="./chroma_db"  # Persists data locally
    )
    print(f"Vector store created with {len(chunks)} documents\n")
    return vector_store

def initialize_rag():
    """Initialize the RAG system."""
    global vector_store, retriever, llm
    
    print("=" * 50)
    print("Initializing RAG System...")
    print("=" * 50 + "\n")
    
    # Load PDFs
    docs = load_pdfs_from_folder("documents")
    
    if not docs:
        print("No documents to process. Please add PDFs to the 'documents' folder.")
        vector_store = None
        retriever = None
        return
    
    # Split documents
    chunks = split_documents(docs)
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500)
    
    print("RAG System Ready!\n")

def query_rag(query: str) -> tuple:
    """Query the RAG system and get an answer."""
    if retriever is None or llm is None:
        raise ValueError("RAG system not initialized. Add PDFs to the 'documents' folder and restart.")
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Create prompt
    template = """You are a helpful assistant who knows the Kenya constitution. You must answer questions based on the provided context and ensure you quote the exact Act you retrieved the information from.
    
Context:
{context}

Question: {query}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format context from retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get answer from LLM
    response = llm.invoke(prompt.format(context=context, query=query))
    
    # Extract source information
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    return response.content, sources

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    initialize_rag()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG PDF Query API",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if vector_store is None:
        return {
            "status": "warning",
            "message": "RAG system not initialized. No PDFs found in 'documents' folder."
        }
    return {
        "status": "healthy",
        "message": "RAG system is ready"
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system."""
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Add PDFs to the 'documents' folder."
        )
    
    try:
        answer, sources = query_rag(request.query)
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=list(set(sources))  # Remove duplicates
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload():
    """Reload documents from the documents folder."""
    initialize_rag()
    return {"message": "RAG system reloaded successfully"}

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("Starting RAG PDF Query Server...")
    print("=" * 50 + "\n")
    print("API Documentation: http://localhost:8000/docs")
    print("Place your PDFs in the 'documents' folder\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)