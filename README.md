🧑‍⚖️ Legal Assistant RAG – Multi-PDF Comparison App
This project is a Retrieval-Augmented Generation (RAG) based Legal Assistant built using LangChain, Streamlit, HuggingFace Embeddings, and OpenAI’s GPT-4 Turbo. It enables users (lawyers, researchers, students) to analyze and compare legal court documents like charge sheets, witness statements, and FIRs.

Features
🔍 Query Legal PDFs: Ask natural-language questions to one or more legal case documents.

📂 Multi-PDF Upload: Upload and work with multiple court documents at once.

🧠 HuggingFace Local Embeddings: Fast, open-source embeddings (no API key required).

🤖 gemini-2.0-flash Powered Answers: High-quality legal summarization and response generation.

🆚 Witness Statement Comparison: Click one button to compare key sections (like witness testimony) across all uploaded PDFs.

📚 Source Chunk Transparency: View exactly what content was used to generate the answer.

💡 Built for Law & Legal Reasoning: Prompt tuned for accuracy and legal formalism.

🛠️ Tech Stack
Component	Tech Used
🧠 LLM	gemini-2.0-flash
📄 Embeddings	HuggingFace all-MiniLM-L6-v2
🧾 Vector Store	LangChain + Chroma
🖥️ UI	Streamlit
📚 Document Loader	LangChain's PDF loader
🧠 Framework	LangChain (chains, retrievers, QA)