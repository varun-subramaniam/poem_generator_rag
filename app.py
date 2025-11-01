import streamlit as st
import chains
from vectordb_config import store_pdf_in_vectordb


def poem_generator_app():
    """
    Generates Poem Generator App with Streamlit, providing user input and displaying output.
    """
    st.sidebar.title('Poem Generator')

    section = st.sidebar.radio("Choose", ("Generate with RAG", "Indexing for RAG"))

    if section == "Generate with RAG":

        st.title("Lets generate a poem ! 👋")

        with st.form("poem_generator"):
            topic = st.text_input("Enter a topic for the poem:")
            submitted = st.form_submit_button("Submit")

            if submitted:
                response = chains.generate_poem_rag_chain(topic)
                st.info(response)

        with st.form("qa_answering"):
            question = st.text_input("Ask a question to know about Sri Eshwar placement coordinator")
            qa_submitted = st.form_submit_button("Get Answer")

            if qa_submitted:
                answer = chains.generate_qa_rag_Chain(question)
                st.success(answer)

    elif section == "Indexing for RAG":
        st.title("RAG File Ingestion")
        st.write("This section will allow you to upload files for RAG indexing.")
        # File upload and indexing logic would go here

        uploaded_file = st.file_uploader("Upload your poem file")

        if uploaded_file is not None:

            store_pdf_in_vectordb(uploaded_file, collection_name = "poem_generator_docs")
            st.success(f"File '{uploaded_file.name}' uploaded and indexed successfully!")

        qa_file = st.file_uploader("Upload your file")

        if qa_file is not None:
            store_pdf_in_vectordb(qa_file, collection_name = "qa_docs")
            st.success(f"File '{qa_file.name}' uploaded and indexed successfully!")





poem_generator_app()
