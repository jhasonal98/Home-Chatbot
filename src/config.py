"""
Configuration file for the Home Chatbot application.
Centralized configuration for models, paths, and hyperparameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs" / "house_details.md"
VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore"

# Embedding model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, fast, accurate
EMBEDDING_DIMENSION = 384

# LLM configuration
# HuggingFace Inference API
HF_MODEL_ID = "google/flan-t5-large"  # Free tier, excellent for Q&A tasks
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Set in Streamlit secrets

# RAG configuration
CHUNK_SIZE = 500  # Words per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
TOP_K_RETRIEVAL = 3  # Number of chunks to retrieve for context

# LLM generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.95

# Application settings
APP_TITLE = "🏠 Home Chatbot - 19th Floor 2BHK"
APP_DESCRIPTION = "Ask questions about this beautiful 2BHK apartment on the 19th floor with a west-facing balcony!"

# Room sections for sidebar navigation
ROOM_SECTIONS = {
    "🏠 Home Overview": "overview",
    "🛋️ Living Room": "living_room",
    "🍳 Kitchen & Dining": "kitchen_dining",
    "🛏️ Master Bedroom": "master_bedroom",
    "🛏️ Second Bedroom": "second_bedroom",
    "🌅 Balcony": "balcony",
    "🚿 Washrooms": "washrooms",
    "💬 Q&A Chat": "qa_chat",
}

# Pre-written descriptions for structured sections
SECTION_DESCRIPTIONS = {
    "overview": {
        "title": "🏠 Welcome to 19th Floor 2BHK Apartment",
        "content": """This is a beautifully designed 2 Bedroom, Hall, and Kitchen (2BHK) apartment 
        located on the 19th floor of a modern high-rise complex. The apartment features excellent natural light, 
        reduced street noise, and panoramic views. The floor plan optimizes space with an open-plan living and dining area, 
        two private bedrooms, and two washrooms (one attached to the master bedroom and one common). 
        Every corner is designed with purpose and functionality."""
    },
    "living_room": {
        "title": "🛋️ Living Room (8 × 14 ft)",
        "content": """The living room is the heart of the apartment, measuring 8 feet wide by 14 feet long. 
        It features a striking blue sofa as the focal point, paired with a small center table for convenience. 
        A tall shoe case with capacity for 30 pairs keeps the entryway organized. 
        A custom 5 × 13 feet carpet covers the floor, adding warmth and acoustic dampening."""
    },
    "kitchen_dining": {
        "title": "🍳 Open Kitchen & Dining Space",
        "content": """The kitchen is an expansive, open-concept space seamlessly connected to the living room. 
        This "kitchen cum dining" design enhances the sense of space, allowing light and conversation to flow freely. 
        The kitchen features modern appliances, extensive countertop space, and a dedicated dining area integrated within. 
        The open architecture makes the entire apartment feel airy and interconnected."""
    },
    "master_bedroom": {
        "title": "🛏️ Master Bedroom with Attached Washroom",
        "content": """The master bedroom is a private sanctuary with a comfortable bed, dedicated work desk, 
        and ergonomic chair—perfect for remote work. A distinctive green-colored almirah (wardrobe) provides storage 
        while adding character to the room. The attached en-suite washroom offers complete privacy and convenience 
        to primary occupants without needing to access common areas."""
    },
    "second_bedroom": {
        "title": "🛏️ Second Bedroom",
        "content": """The secondary bedroom is versatile, suitable as a guest room, child's room, or home office. 
        It includes a comfortable bed and separate work desk, allowing multiple residents to work simultaneously. 
        The room's defining feature is direct glass door access to the apartment's exclusive west-facing balcony."""
    },
    "balcony": {
        "title": "🌅 West-Facing Balcony",
        "content": """Accessible from the second bedroom via a glass door, this small but spectacular balcony 
        faces west offering unobstructed panoramic sky views. Located on the 19th floor, it provides direct 
        afternoon sun and spectacular sunset viewing opportunities. The high altitude ensures steady cross-breezes, 
        making it perfect for morning coffee or evening relaxation while overlooking the bustling city."""
    },
    "washrooms": {
        "title": "🚿 Washrooms",
        "content": """The apartment features two well-appointed washrooms: 
        (1) An attached en-suite washroom in the master bedroom for privacy of primary occupants, 
        (2) A centrally located common washroom accessible from living areas for guests and second bedroom users. 
        Both are equipped with standard modern sanitary ware. The dual washroom setup eliminates bottlenecks 
        during busy morning routines."""
    },
    "qa_chat": {
        "title": "💬 Ask Questions About the Apartment",
        "content": """Use the chat box below to ask any questions about this apartment! 
        The chatbot will search through the apartment's description and provide detailed answers 
        about specific features, dimensions, amenities, or anything else you'd like to know."""
    }
}

print("✅ Configuration loaded successfully")
