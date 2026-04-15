"""
RAG (Retrieval-Augmented Generation) Pipeline
Handles document loading, embedding, FAISS indexing, and retrieval.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple
import json

from src.config import (
    DOCS_PATH, VECTORSTORE_PATH, EMBEDDING_MODEL, 
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
)


def get_document_class():
    """Lazy import Document class."""
    try:
        from langchain.schema import Document
        return Document
    except ImportError:
        try:
            from langchain_core.documents import Document
            return Document
        except ImportError:
            raise RuntimeError("Could not import Document from LangChain")


def get_text_splitter_class():
    """Lazy import RecursiveCharacterTextSplitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter
        except ImportError:
            raise RuntimeError("Could not import RecursiveCharacterTextSplitter from LangChain")


def get_faiss_class():
    """Lazy import FAISS."""
    try:
        from langchain_community.vectorstores import FAISS
        return FAISS
    except ImportError:
        try:
            from langchain.vectorstores import FAISS
            return FAISS
        except ImportError:
            raise RuntimeError("Could not import FAISS from LangChain")


def get_embeddings_class():
    """Lazy import HuggingFaceEmbeddings."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings
        except ImportError:
            raise RuntimeError("Could not import HuggingFaceEmbeddings from LangChain")


class RAGPipeline:
    """
    Manages document loading, embedding, and FAISS indexing for RAG.
    """
    
    def __init__(self):
        """Initialize the RAG pipeline (lightweight, fast initialization - imports on demand)."""
        self.embeddings = None
        self.vectorstore = None
        self.docs = []
        self.initialization_error = None
        
        print("✅ RAG Pipeline initialized (modules loaded on-demand)")
    
    def _ensure_embeddings(self):
        """Lazy-load embeddings model on first use."""
        if self.embeddings is not None:
            return
        
        try:
            print(f"🔄 Loading embeddings model: {EMBEDDING_MODEL} (this may take 30-60 seconds on first run)...")
            HuggingFaceEmbeddings = get_embeddings_class()
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={"normalize_embeddings": True}
            )
            print(f"✅ Embeddings model loaded: {EMBEDDING_MODEL}")
        except Exception as e:
            error_msg = f"Could not load embeddings model: {str(e)}"
            self.initialization_error = error_msg
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def load_documents(self, doc_path: str = None) -> List:
        """
        Load documents from markdown file.
        
        Args:
            doc_path: Path to markdown file (default: DOCS_PATH from config)
        
        Returns:
            List of Document objects
        """
        if doc_path is None:
            doc_path = DOCS_PATH
        
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        Document = get_document_class()
        # Create document with metadata
        doc = Document(
            page_content=text,
            metadata={"source": doc_path, "type": "house_description"}
        )
        
        self.docs = [doc]
        print(f"✅ Loaded document: {doc_path} ({len(text)} characters)")
        return self.docs
    
    def create_chunks(self, docs: List = None) -> List:
        """
        Split documents into chunks for better retrieval.
        
        Args:
            docs: List of documents to chunk (default: self.docs)
        
        Returns:
            List of chunked documents
        """
        if docs is None:
            docs = self.docs
        
        if not docs:
            raise ValueError("No documents to chunk. Call load_documents() first.")
        
        RecursiveCharacterTextSplitter = get_text_splitter_class()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        return chunks
    
    def create_or_load_vectorstore(self, force_rebuild: bool = False):
        """
        Create FAISS vectorstore or load existing one.
        
        Args:
            force_rebuild: If True, rebuild index even if it exists
        
        Returns:
            FAISS vectorstore object
        """
        if self.initialization_error:
            raise RuntimeError(f"RAG Pipeline initialization error: {self.initialization_error}")
        
        # Lazy-load embeddings if not already loaded
        self._ensure_embeddings()
        
        if not self.embeddings:
            raise ValueError("Embeddings model not initialized")
        
        FAISS = get_faiss_class()
        vectorstore_path = str(VECTORSTORE_PATH / "index")
        
        # Check if existing vectorstore can be loaded
        if not force_rebuild and (VECTORSTORE_PATH / "index.faiss").exists():
            print("📚 Loading existing FAISS index...")
            try:
                self.vectorstore = FAISS.load_local(
                    str(VECTORSTORE_PATH),
                    self.embeddings,
                    index_name="index",
                    allow_dangerous_deserialization=True
                )
                print("✅ FAISS index loaded successfully")
                return self.vectorstore
            except Exception as e:
                print(f"⚠️ Warning: Could not load existing index: {e}. Rebuilding...")
        
        # Create new vectorstore
        print("🔨 Building FAISS index from documents...")
        
        # Load and chunk documents
        if not self.docs:
            self.load_documents()
        
        chunks = self.create_chunks()
        
        try:
            # Create FAISS vectorstore
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # Save vectorstore
            VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(VECTORSTORE_PATH), index_name="index")
            
            print(f"✅ FAISS index created and saved to {VECTORSTORE_PATH}")
            return self.vectorstore
        except Exception as e:
            error_msg = f"Failed to create FAISS index: {str(e)}"
            self.initialization_error = error_msg
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def retrieve(self, query: str, k: int = None) -> Tuple[List[str], List[dict]]:
        """
        Retrieve relevant chunks from FAISS index.
        
        Args:
            query: User question or query string
            k: Number of chunks to retrieve (default: TOP_K_RETRIEVAL from config)
        
        Returns:
            Tuple of (context_chunks, metadata)
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_or_load_vectorstore() first.")
        
        if k is None:
            k = TOP_K_RETRIEVAL
        
        # Retrieve documents
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            context_chunks = [doc.page_content for doc in docs]
            metadata = [doc.metadata for doc in docs]
            
            print(f"✅ Retrieved {len(docs)} relevant chunks for query: '{query}'")
            return context_chunks, metadata
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return [], []
    
    def get_context_string(self, query: str, k: int = None) -> str:
        """
        Get retrieved context as a formatted string.
        
        Args:
            query: User question
            k: Number of chunks to retrieve
        
        Returns:
            Formatted context string
        """
        context_chunks, _ = self.retrieve(query, k=k)
        
        if not context_chunks:
            return "No relevant information found in the knowledge base."
        
        context = "\n\n---\n\n".join(context_chunks)
        return context


# Global RAG pipeline instance
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or initialize the global RAG pipeline instance.
    """
    global _rag_pipeline
    
    if _rag_pipeline is None:
        print("🔄 Initializing RAG pipeline...")
        _rag_pipeline = RAGPipeline()
        
        # Check for initialization errors
        if _rag_pipeline.initialization_error:
            print(f"⚠️ RAG Pipeline Error: {_rag_pipeline.initialization_error}")
            return _rag_pipeline
        
        try:
            print("📄 Loading documents...")
            _rag_pipeline.load_documents()
            print("🔨 Creating FAISS index...")
            _rag_pipeline.create_or_load_vectorstore()
            print("✅ RAG pipeline ready!")
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            _rag_pipeline.initialization_error = str(e)
    
    return _rag_pipeline


def retrieve_context(query: str, k: int = None) -> Tuple[List[str], List[dict]]:
    """
    Convenience function to retrieve context without manual pipeline initialization.
    """
    pipeline = get_rag_pipeline()
    return pipeline.retrieve(query, k=k)


def get_context_string(query: str, k: int = None) -> str:
    """
    Convenience function to get formatted context string.
    """
    pipeline = get_rag_pipeline()
    return pipeline.get_context_string(query, k=k)
