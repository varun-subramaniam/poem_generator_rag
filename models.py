from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def create_chat_groq_model(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
):
    """
    Creates and returns a configured instance of the ChatGroq model.

    Args:
        model -> str: The model to use (default: "llama-3.1-8b-instant").
        temperature -> float: Sampling temperature for randomness (default: 0).
        max_tokens -> int or None: Maximum number of tokens to generate (default: None).
        timeout -> int or None: Timeout for requests in seconds (default: None).
        max_retries -> int: Number of retries on request failures (default: 2).

    Returns:
        ChatGroq: Configured ChatGroq model instance
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        cache=False
    )

def create_huggingface_model(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto"
):
    """
    Creates and returns a configured instance of the ChatHuggingFace model.

    Args:
        repo_id -> str: The Hugging Face model repository ID (default: "openai/gpt-oss-20b").
        task -> str: The task type for the model (default: "text-generation").
        max_new_tokens -> int: Maximum number of new tokens to generate (default: 512).
        do_sample -> bool: Whether to use sampling for generation (default: False).
        repetition_penalty -> float: Penalty for repetition in generated text (default: 1.03).
        provider -> str: Provider selection, "auto" lets Hugging Face choose (default: "auto").

    Returns:
        ChatHuggingFace: Configured ChatHuggingFace model instance
    """
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task=task,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        provider=provider
    )
    
    return ChatHuggingFace(llm=llm)

def create_gemini_embedding_model():
    """
    Creates and returns a configured instance of the Google Gemini embeddings model.

    Returns:
        GeminiEmbeddings: Configured Google Gemini embedding model
    """
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def create_hugging_face_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Creates and returns a configured instance of the huggingface embeddings model.

    Args:
        model_name -> str: The model to use (default: "sentence-transformers/all-MiniLM-L6-v2").

    Returns:
        HuggingFaceEmbeddings: Configured HuggingFaceEmbeddings model instance
    """
    return HuggingFaceEmbeddings(model_name=model_name)