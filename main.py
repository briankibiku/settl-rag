# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from fastapi.middleware.cors import CORSMiddleware


# # Load environment variables
# load_dotenv()

# # Verify API key exists
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not set in .env file")

# # Initialize FastAPI
# app = FastAPI(title="RAG PDF Query API", version="1.0.0")

# # Add CORS middleware - Allow all origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],                    # Allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],                    # Allow all HTTP methods
#     allow_headers=["*"],                    # Allow all headers
# )

# # Global variables for vector store
# vector_store = None
# retriever = None
# llm = None

# # Pydantic models for API
# class QueryRequest(BaseModel):
#     query: str

# class QueryResponse(BaseModel):
#     query: str
#     answer: str
#     sources: list

# # ==================== RAG Functions ====================

# def load_pdfs_from_folder(folder_path: str = "documents"):
#     """Load all PDFs from a folder."""
#     print(f"Loading PDFs from {folder_path}...")
#     docs = []
    
#     folder = Path(folder_path)
#     if not folder.exists():
#         print(f"Creating {folder} folder...")
#         folder.mkdir(exist_ok=True)
#         print(f"Please place your PDF files in the '{folder}' folder")
#         return docs
    
#     pdf_files = list(folder.glob("*.pdf"))
    
#     if not pdf_files:
#         print(f"No PDF files found in {folder_path}")
#         return docs
    
#     for pdf_file in pdf_files:
#         print(f"  Loading: {pdf_file.name}")
#         loader = PyPDFLoader(str(pdf_file))
#         docs.extend(loader.load())
    
#     print(f"Total pages loaded: {len(docs)}\n")
#     return docs

# def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200):
#     """Split documents into chunks."""
#     print("Splitting documents into chunks...")
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = splitter.split_documents(docs)
#     print(f"Created {len(chunks)} chunks\n")
#     return chunks

# def create_vector_store(chunks: list):
#     """Create and store embeddings in Chroma."""
#     print("Creating embeddings and storing in Chroma...")
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
#     # Create Chroma vector store
#     vector_store = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         collection_name="pdf-documents",
#         persist_directory="./chroma_db"  # Persists data locally
#     )
#     print(f"Vector store created with {len(chunks)} documents\n")
#     return vector_store

# def initialize_rag():
#     """Initialize the RAG system."""
#     global vector_store, retriever, llm
    
#     print("=" * 50)
#     print("Initializing RAG System...")
#     print("=" * 50 + "\n")
    
#     # Load PDFs
#     docs = load_pdfs_from_folder("documents")
    
#     if not docs:
#         print("No documents to process. Please add PDFs to the 'documents' folder.")
#         vector_store = None
#         retriever = None
#         return
    
#     # Split documents
#     chunks = split_documents(docs)
    
#     # Create vector store
#     vector_store = create_vector_store(chunks)
    
#     # Create retriever
#     retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
#     # Initialize LLM
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500)
    
#     print("RAG System Ready!\n")

# def query_rag(query: str) -> tuple:
#     """Query the RAG system and get an answer."""
#     if retriever is None or llm is None:
#         raise ValueError("RAG system not initialized. Add PDFs to the 'documents' folder and restart.")
    
#     # Retrieve relevant documents
#     docs = retriever.invoke(query)
    
#     # Create prompt
#     template = """
#         You are a helpful legal research assistant specialized in Kenyan road traffic accident personal injury cases. You must answer strictly based on the provided context retrieved from the vector database and must not rely on outside knowledge or assumptions.

#         Your task is to identify court cases involving non-fatal road traffic accident injuries that are medically similar to the injuries provided in the question. You must extract and summarize only the general damages awarded for pain, suffering, and loss of amenities.

#         STRICT RULES:
#         - Use only the information explicitly present in the provided context.
#         - Exclude any case involving death, deceased estates, fatal injuries, loss of dependency, or claims under the Law Reform Act or Fatal Accidents Act.
#         - If a case’s fatality status is unclear, exclude it.
#         - Do not include special damages, future medical expenses, loss of earnings, costs, or interest.
#         - Only consider judgments from Kenyan courts and awards in Kenya Shillings (KES).
#         - Do not guess, estimate, or infer figures not clearly stated in the context.

#         Context:
#         {context}

#         Question:
#         {query}

#         Answer:
#         - Injury Summary (normalized from the question)
#         - Comparable Non-Fatal Cases:
#         • Case name, year, court  
#         • Injuries sustained  
#         • General damages awarded (KES)
#         - Observed Award Range (min / max / median if available)
#         - Brief Notes explaining injury severity and factors influencing the awards

#         Damages estimate:
#         - Maximam general damages likely to be awarded based on comparable cases.
#         - Minimum general damages likely to be awarded based on comparable cases.

#         If fewer than two valid comparable non-fatal cases are found, clearly state: “Insufficient comparable non-fatal cases found based on the provided context.”

#     """
    
#     prompt = ChatPromptTemplate.from_template(template)
    
#     # Format context from retrieved docs
#     context = "\n\n".join([doc.page_content for doc in docs])
    
#     # Get answer from LLM
#     response = llm.invoke(prompt.format(context=context, query=query))
    
#     # Extract source information
#     sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
#     return response.content, sources

# # ==================== API Endpoints ====================

# @app.on_event("startup")
# async def startup_event():
#     """Initialize RAG system on startup."""
#     initialize_rag()

# @app.get("/")
# async def root():
#     """Root endpoint."""
#     return {
#         "message": "RAG PDF Query API",
#         "endpoints": {
#             "query": "POST /query",
#             "health": "GET /health"
#         }
#     }

# @app.get("/health")
# async def health():
#     """Health check endpoint."""
#     if vector_store is None:
#         return {
#             "status": "warning",
#             "message": "RAG system not initialized. No PDFs found in 'documents' folder."
#         }
#     return {
#         "status": "healthy",
#         "message": "RAG system is ready"
#     }

# @app.post("/query", response_model=QueryResponse)
# async def query_endpoint(request: QueryRequest):
#     """Query the RAG system."""
#     if vector_store is None:
#         raise HTTPException(
#             status_code=503,
#             detail="RAG system not initialized. Add PDFs to the 'documents' folder."
#         )
    
#     try:
#         answer, sources = query_rag(request.query)
#         return QueryResponse(
#             query=request.query,
#             answer=answer,
#             sources=list(set(sources))  # Remove duplicates
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/reload")
# async def reload():
#     """Reload documents from the documents folder."""
#     initialize_rag()
#     return {"message": "RAG system reloaded successfully"}

# # ==================== Main ====================

# if __name__ == "__main__":
#     import uvicorn
#     print("\n" + "=" * 50)
#     print("Starting RAG PDF Query Server...")
#     print("=" * 50 + "\n")
#     print("API Documentation: http://localhost:8000/docs")
#     print("Place your PDFs in the 'documents' folder\n")
    
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
import re
import requests
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
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for vector store
vector_store = None
retriever = None
llm = None
kenyalaw_store = None  # Separate Chroma collection for Kenya Law cases

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str

class SimilarCase(BaseModel):
    title: str
    url: str
    score: float
    case_number: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    similar_cases: List[SimilarCase] = []
    case_urls: List[str] = []
    min_damages: Optional[int] = None
    max_damages: Optional[int] = None

# ==================== Kenya Law API Functions ====================

def fetch_kenya_law_cases(search_query: str = "road traffic accident", max_pages: int = 10) -> List[Dict]:
    """
    Fetch cases from Kenya Law Reports API.
    """
    all_cases = []
    base_url = "https://new.kenyalaw.org/search/api/documents/"
    
    print(f"Fetching cases from Kenya Law Reports for query: '{search_query}'...")
    
    for page in range(1, max_pages + 1):
        try:
            params = {
                "search": search_query,
                "page": page,
                "ordering": "-score",
                "nature": "Judgment",
                "facet": ["nature", "court", "year", "registry", "outcome"],
                "limit": 10
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                print(f"No more results at page {page}")
                break
            
            all_cases.extend(results)
            print(f"  Fetched page {page}: {len(results)} cases")
            
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    print(f"Total cases fetched: {len(all_cases)}\n")
    return all_cases

def store_cases_in_chroma(cases: List[Dict]):
    """
    Store Kenya Law cases in Chroma vector database.
    """
    global kenyalaw_store
    
    print("Storing Kenya Law cases in Chroma...")
    
    documents = []
    
    for case in cases:
        try:
            case_id = case.get("id", "")
            title = case.get("title", "")
            citation = case.get("citation", "")
            case_number = case.get("case_number", [])
            judges = case.get("judges", [])
            year = case.get("year", "")
            court = case.get("court", "")
            expression_frbr_uri = case.get("expression_frbr_uri", "")
            
            # Create searchable text content
            content_parts = [
                f"Title: {title}",
                f"Citation: {citation}",
                f"Case Number: {', '.join(case_number) if case_number else 'N/A'}",
                f"Court: {court}",
                f"Year: {year}",
                f"Judges: {', '.join(judges) if judges else 'N/A'}",
            ]
            
            # Add highlight content if available
            if "highlight" in case and "content" in case["highlight"]:
                highlights = case["highlight"]["content"]
                # Remove HTML marks
                clean_highlights = [re.sub(r'</?mark>', '', h) for h in highlights]
                content_parts.append(f"Relevant excerpts: {' '.join(clean_highlights)}")
            
            content = "\n".join(content_parts)
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "id": case_id,
                    "title": title,
                    "citation": citation,
                    "case_number": ", ".join(case_number) if case_number else "",
                    "year": year,
                    "court": court,
                    "judges": ", ".join(judges) if judges else "",
                    "expression_frbr_uri": expression_frbr_uri,
                    "url": construct_case_url(expression_frbr_uri),
                    "source": "KenyaLaw",
                    "score": case.get("_score", 0)
                }
            )
            
            documents.append(doc)
            
        except Exception as e:
            print(f"Error processing case {case.get('id')}: {e}")
    
    if documents:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create separate Chroma collection for Kenya Law cases
        kenyalaw_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="kenyalaw-cases",
            persist_directory="./chroma_db"
        )
        
        print(f"Stored {len(documents)} Kenya Law cases in Chroma\n")
    else:
        print("No documents to store in Chroma\n")

def construct_case_url(expression_frbr_uri: str) -> str:
    """
    Construct full Kenya Law URL from expression_frbr_uri.
    """
    if not expression_frbr_uri:
        return ""
    return f"https://new.kenyalaw.org{expression_frbr_uri}"

def search_similar_cases_in_chroma(case_names: List[str], query: str) -> List[SimilarCase]:
    """
    Search for similar cases in Chroma based on case names and query.
    """
    if not kenyalaw_store:
        return []
    
    similar_cases = []
    seen_urls = set()
    
    # Search using the original query to find relevant cases
    try:
        # Use similarity search to find related cases
        results = kenyalaw_store.similarity_search_with_score(query, k=10)
        
        for doc, score in results:
            url = doc.metadata.get("url", "")
            
            # Avoid duplicates
            if url and url not in seen_urls:
                seen_urls.add(url)
                
                similar_cases.append(SimilarCase(
                    title=doc.metadata.get("title", ""),
                    url=url,
                    score=round(1 - (score / 2), 2),  # Normalize score (lower distance = higher similarity)
                    case_number=doc.metadata.get("case_number", ""),
                    year=doc.metadata.get("year"),
                    court=doc.metadata.get("court", "")
                ))
        
        # Also search by case names mentioned in response
        for case_name in case_names:
            if case_name:
                name_results = kenyalaw_store.similarity_search(case_name, k=2)
                
                for doc in name_results:
                    url = doc.metadata.get("url", "")
                    
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        
                        similar_cases.append(SimilarCase(
                            title=doc.metadata.get("title", ""),
                            url=url,
                            score=0.85,  # High score for name match
                            case_number=doc.metadata.get("case_number", ""),
                            year=doc.metadata.get("year"),
                            court=doc.metadata.get("court", "")
                        ))
        
    except Exception as e:
        print(f"Error searching cases in Chroma: {e}")
    
    # Sort by score descending and limit to top 10
    similar_cases.sort(key=lambda x: x.score, reverse=True)
    return similar_cases[:10]

def extract_case_names_from_response(answer: str) -> List[str]:
    """
    Extract case names from the LLM response.
    """
    # Pattern to match case names like "Name v Name, Year, Court"
    pattern = r'([A-Z][a-zA-Z\s&]+\s+v\s+[A-Z][a-zA-Z\s&]+(?:\s+&\s+[A-Za-z\s]+)*),?\s+(\d{4})'
    matches = re.findall(pattern, answer)
    
    case_names = [match[0].strip() for match in matches]
    return case_names

def extract_damages_from_response(answer: str) -> tuple:
    """
    Extract minimum and maximum damages from the response.
    """
    min_damages = None
    max_damages = None
    
    # Pattern to match damages amounts like "KES 1,000,000" or "1000000"
    min_pattern = r'[Mm]inimum[:\s]+(?:KES\s+)?([0-9,]+)'
    max_pattern = r'[Mm]aximum[:\s]+(?:KES\s+)?([0-9,]+)'
    
    min_match = re.search(min_pattern, answer)
    max_match = re.search(max_pattern, answer)
    
    if min_match:
        min_damages = int(min_match.group(1).replace(',', ''))
    
    if max_match:
        max_damages = int(max_match.group(1).replace(',', ''))
    
    return min_damages, max_damages

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
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf-documents",
        persist_directory="./chroma_db"
    )
    print(f"Vector store created with {len(chunks)} documents\n")
    return vector_store

def initialize_rag():
    """Initialize the RAG system and fetch Kenya Law cases."""
    global vector_store, retriever, llm, kenyalaw_store
    
    print("=" * 50)
    print("Initializing RAG System...")
    print("=" * 50 + "\n")
    
    # Load PDFs
    docs = load_pdfs_from_folder("documents")
    
    if not docs:
        print("No documents to process. Please add PDFs to the 'documents' folder.")
        vector_store = None
        retriever = None
    else:
        # Split documents
        chunks = split_documents(docs)
        
        # Create vector store
        vector_store = create_vector_store(chunks)
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500)
        
        print("RAG System Ready!\n")
    
    # Fetch and store Kenya Law cases in Chroma
    cases = fetch_kenya_law_cases("road traffic accident", max_pages=10)
    if cases:
        store_cases_in_chroma(cases)

def query_rag(query: str) -> dict:
    """Query the RAG system and get an answer with Kenya Law URLs."""
    if retriever is None or llm is None:
        raise ValueError("RAG system not initialized. Add PDFs to the 'documents' folder and restart.")
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Create prompt
    template = """
        You are a helpful legal research assistant specialized in Kenyan road traffic accident personal injury cases. You must answer strictly based on the provided context retrieved from the vector database and must not rely on outside knowledge or assumptions.

        Your task is to identify court cases involving non-fatal road traffic accident injuries that are medically similar to the injuries provided in the question. You must extract and summarize only the general damages awarded for pain, suffering, and loss of amenities.

        STRICT RULES:
        - Use only the information explicitly present in the provided context.
        - Exclude any case involving death, deceased estates, fatal injuries, loss of dependency, or claims under the Law Reform Act or Fatal Accidents Act.
        - If a case's fatality status is unclear, exclude it.
        - Do not include special damages, future medical expenses, loss of earnings, costs, or interest.
        - Only consider judgments from Kenyan courts and awards in Kenya Shillings (KES).
        - Do not guess, estimate, or infer figures not clearly stated in the context.

        Context:
        {context}

        Question:
        {query}

        Answer:
        - Injury Summary (normalized from the question)
        - Comparable Non-Fatal Cases:
        • Case name, year, court  
        • Injuries sustained  
        • General damages awarded (KES)
        - Observed Award Range (min / max / median if available)
        - Brief Notes explaining injury severity and factors influencing the awards

        Damages estimate:
        - Maximum general damages likely to be awarded based on comparable cases.
        - Minimum general damages likely to be awarded based on comparable cases.

        If fewer than two valid comparable non-fatal cases are found, clearly state: "Insufficient comparable non-fatal cases found based on the provided context."
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format context from retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get answer from LLM
    response = llm.invoke(prompt.format(context=context, query=query))
    answer = response.content
    
    # Extract source information
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    # Extract case names from response
    case_names = extract_case_names_from_response(answer)
    
    # Search for similar cases in Chroma
    similar_cases = search_similar_cases_in_chroma(case_names, query)
    
    # Extract case URLs
    case_urls = [case.url for case in similar_cases if case.url]
    
    # Extract damages range
    min_damages, max_damages = extract_damages_from_response(answer)
    
    return {
        "query": query,
        "answer": answer,
        "sources": list(set(sources)),
        "similar_cases": [case.dict() for case in similar_cases],
        "case_urls": case_urls,
        "min_damages": min_damages,
        "max_damages": max_damages
    }

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    initialize_rag()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG PDF Query API with Kenya Law Integration",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "reload": "POST /reload",
            "refresh_cases": "POST /refresh-cases"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if vector_store is None:
        status = "warning"
        message = "RAG system not initialized. No PDFs found in 'documents' folder."
    else:
        status = "healthy"
        message = "RAG system is ready"
    
    kenyalaw_status = "loaded" if kenyalaw_store else "not loaded"
    
    return {
        "status": status,
        "message": message,
        "kenyalaw_cases": kenyalaw_status
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
        result = query_rag(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload():
    """Reload documents from the documents folder."""
    initialize_rag()
    return {"message": "RAG system reloaded successfully"}

@app.post("/refresh-cases")
async def refresh_cases():
    """Refresh Kenya Law cases in Chroma."""
    try:
        cases = fetch_kenya_law_cases("road traffic accident", max_pages=10)
        if cases:
            store_cases_in_chroma(cases)
            return {
                "message": "Kenya Law cases refreshed successfully",
                "cases_cached": len(cases)
            }
        else:
            return {
                "message": "No cases fetched",
                "cases_cached": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("Starting RAG PDF Query Server...")
    print("=" * 50 + "\n")
    print("API Documentation: http://localhost:8000/docs")
    print("Place your PDFs in the 'documents' folder\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)