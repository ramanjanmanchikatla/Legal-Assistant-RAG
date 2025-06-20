import streamlit as st
import os
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()




# Set page config
st.set_page_config(page_title="🧑‍⚖️ Legal Assistant - Multi PDF Comparator", layout="wide")
st.title("🧑‍⚖️ Legal Assistant RAG ")

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more Legal Case PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Initialize embedding and LLM
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # GPT-4 Turbo with larger context window

    vector_stores = {}
    file_names = []

    with st.spinner("Processing documents and creating vector stores..."):
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            file_name = uploaded_file.name
            file_names.append(file_name)

            # Load and chunk
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)  # Suitable for GPT-4 Turbo
            chunks = splitter.split_documents(docs)

            # Create vector store for each doc
            vector_store = Chroma(
                embedding_function=embedding_model,
                persist_directory=f"chroma_db/{file_name}",
                collection_name=file_name
            )
            vector_store.add_documents(chunks)
            retriever = vector_store.as_retriever()
            vector_stores[file_name] = retriever

    st.success("✅ Documents processed and indexed.")

    # User input for query
    st.markdown("---")
    st.subheader("📌 Ask a legal question to one specific document")
    selected_doc = st.selectbox("Choose document to query:", file_names)
    user_query = st.text_input("Enter your question:")

    if st.button("🔍 Get Answer") and user_query:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_stores[selected_doc],
            chain_type="stuff",
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": user_query})
        st.markdown(f"### 🧠 Answer from *{selected_doc}*:")
        st.write(result["result"])

        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**Chunk {i}:**\n{doc.page_content[:700]}...")

    # Comparison feature
    st.markdown("---")
    st.subheader("🔎 Compare Witness Statements Across All Documents")
    if st.button("🧑‍⚖️ Compare Witness Statements"):
        comparison_query = "metion all the details of the cases?"
        for name in file_names:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_stores[name],
                chain_type="stuff"
            )
            result = qa_chain.invoke({"query": comparison_query})
            st.markdown(f"### 📄 {name}")
            st.write(result["result"])

else:
    st.info("👆 Upload one or more court-related PDFs to begin.")
