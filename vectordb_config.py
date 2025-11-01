from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import models


def initialize_vectordb(collection_name="poem_generator_docs"):

    hugging_face_embeddings = models.create_hugging_face_embedding_model()

    vector_db = Chroma(embedding_function=hugging_face_embeddings, collection_name=collection_name, persist_directory="./chroma_db")

    return vector_db


def store_pdf_in_vectordb(uploaded_file, collection_name):

    vectordb = initialize_vectordb(collection_name)

    file_path = f"data_source/temp_{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectordb.add_documents(splits)
