import streamlit as st
import os
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
load_dotenv()
index_name = "legalai"  # change if desired
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(index_name)
st.set_page_config(page_title="🧑‍⚖️ Legal Assistant - Multi PDF Comparator", layout="wide")
st.title("🧑‍⚖️ Legal Assistant RAG - Compare Legal Documents")

uploaded_files = st.file_uploader("Upload one or more Legal Case PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    vector_stores = {}
    file_names = []

    with st.spinner("Processing documents and creating vector indexes on Pinecone..."):
        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            file_name = uploaded_file.name
            file_names.append(file_name)

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
            chunks = splitter.split_documents(docs)
            vector_store = PineconeVectorStore(
                    index=index,
                    embedding=embedding_model,
                    namespace=file_name  # Use filename as a unique namespace
                )
            vector_store.add_documents(documents=chunks)

            

            retriever = vector_store.as_retriever()
            vector_stores[file_name] = retriever

    st.success("✅ Documents processed and indexed on Pinecone.")

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
