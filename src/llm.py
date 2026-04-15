"""
LLM Integration Module
Handles inference with HuggingFace models for the chatbot.
"""

import os
from typing import Generator

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

from src.config import (
    HF_MODEL_ID, HF_API_TOKEN, MAX_NEW_TOKENS, 
    TEMPERATURE, TOP_P
)


class HuggingFaceInference:
    """
    Wrapper for HuggingFace Inference API using InferenceClient.
    Uses question-answering task for best free-tier compatibility.
    """
    
    def __init__(self, api_token: str = None):
        """
        Initialize HuggingFace inference client.
        
        Args:
            api_token: HuggingFace API token (default: from config/environment)
        """
        self.api_token = api_token or HF_API_TOKEN
        
        if not self.api_token:
            raise ValueError(
                "HuggingFace API token not found. "
                "Please set HF_API_TOKEN in Streamlit secrets or environment."
            )
        
        if InferenceClient is None:
            raise ImportError("huggingface_hub library not found. Install with: pip install huggingface_hub")
        
        self.client = InferenceClient(api_key=self.api_token)
        print(f"✅ HuggingFace Inference initialized")
    
    def answer_question(
        self,
        question: str,
        context: str = ""
    ) -> str:
        """
        Answer a question using HuggingFace question-answering task.
        
        Args:
            question: User's question
            context: Context/background information
        
        Returns:
            Answer to the question
        """
        if not context:
            context = "The user is asking about a 2BHK apartment on the 19th floor."
        
        try:
            response = self.client.question_answering(
                question=question,
                context=context
            )
            
            # Extract answer from response
            if isinstance(response, dict) and "answer" in response:
                return response["answer"]
            else:
                return str(response)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error details: {error_msg}")
            
            if "503" in error_msg or "loading" in error_msg.lower():
                return "⏳ The model is loading. Please try again in a few seconds."
            elif "404" in error_msg:
                return "❌ Model not found."
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                return "❌ Invalid HuggingFace API token."
            else:
                # Return a more informative error
                return f"API Error: {error_msg[:100]}"


class ChatbotLLM:
    """
    High-level chatbot interface for answering house-related questions.
    """
    
    SYSTEM_PROMPT = """Answer questions about a 2BHK apartment on the 19th floor. 
Be helpful, accurate, and concise. If the answer is not in the provided context, say so."""
    
    def __init__(self, api_token: str = None):
        """
        Initialize the chatbot LLM.
        
        Args:
            api_token: HuggingFace API token
        """
        try:
            self.inference = HuggingFaceInference(api_token)
            self.model_ready = True
        except ValueError as e:
            print(f"⚠️ Warning: {e}")
            self.model_ready = False
    
    def create_prompt(self, question: str, context: str = "") -> str:
        """
        Create a prompt for the LLM combining question and context.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
        
        Returns:
            Formatted prompt
        """
        if context:
            prompt = f"""{self.SYSTEM_PROMPT}

Context about the apartment:
{context}

User Question: {question}

Answer:"""
        else:
            prompt = f"""{self.SYSTEM_PROMPT}

User Question: {question}

Answer:"""
        
        return prompt
    
    def answer_question(self, question: str, context: str = "") -> str:
        """
        Answer a question about the apartment using the LLM.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
        
        Returns:
            Answer from the LLM
        """
        if not self.model_ready:
            return "❌ LLM model is not available. Please check your HuggingFace API token in secrets."
        
        # Use the question-answering task
        answer = self.inference.answer_question(question, context)
        
        return answer
    
    def answer_question_streaming(
        self,
        question: str,
        context: str = ""
    ) -> Generator[str, None, None]:
        """
        Answer a question with streaming output (character by character).
        Note: HuggingFace API doesn't natively support streaming,
        so this yields the full response character by character.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
        
        Yields:
            Characters of the answer
        """
        answer = self.answer_question(question, context)
        
        # Yield answer character by character for streaming effect
        for char in answer:
            yield char


# Global LLM instance
_llm = None


def get_llm(api_token: str = None) -> ChatbotLLM:
    """
    Get or initialize the global LLM instance.
    """
    global _llm
    
    if _llm is None:
        _llm = ChatbotLLM(api_token)
    
    return _llm


def answer_question(question: str, context: str = "") -> str:
    """
    Convenience function to answer questions without manual LLM initialization.
    """
    llm = get_llm()
    return llm.answer_question(question, context)
