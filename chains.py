from langchain_core.output_parsers import StrOutputParser

import models
import prompts
from vectordb_config import initialize_vectordb


def generate_poem_chain(topic):
    """
    Generate Poem using basic prompt LLM chain

    Args:
        topic - topic for the poem

    Returns:
        response.content -> str
    """

    llm = models.create_huggingface_model()

    prompt_template = prompts.poem_generator_prompt()

    chain = prompt_template | llm

    response = chain.invoke({
        "topic": topic
    })
    return response.content


def generate_poem_rag_chain(topic):
    db = initialize_vectordb(collection_name="poem_generator_docs")

    retriever = db.as_retriever()

    retriever_results = retriever.invoke(topic)

    for split in retriever_results:
        print(split)
        print("\n---\n")

    llm = models.create_chat_groq_model()

    rag_chain = prompts.poem_generator_rag_prompt() | llm | StrOutputParser()

    response = rag_chain.invoke({
        "topic": topic,
        "context": "\n\n".join(doc.page_content for doc in retriever_results)
    })

    return response

def generate_qa_rag_Chain(question):
    db = initialize_vectordb(collection_name="qa_docs")

    retriever = db.as_retriever()

    retriever_results = retriever.invoke(question)

    for split in retriever_results:
        print(split)
        print("\n---\n")

    llm = models.create_chat_groq_model()

    rag_chain = prompts.qa_prompt_from_rag() | llm | StrOutputParser()

    response = rag_chain.invoke({
        "question": question,
        "context": "\n\n".join(doc.page_content for doc in retriever_results)
    })

    return response