import streamlit as st
from rag_pipeline import RAGPipeline

st.title("Insurance Carrier RAG System")
st.write("Use this interface to answer questions about carrier appetites.")

# Initialize our pipeline
rag = RAGPipeline(
    json_path="./data/guide_novella.json",
    vector_db_path="./chroma_db",
    openai_api_key=None  # or your actual key if using OpenAI
)

# Buttons to create/load the vector DB
if st.button("Initialize Knowledge Base"):
    with st.spinner("Processing PDFs and creating vector database..."):
        rag.create_vector_db()
    st.success("Vector DB initialized!")

if st.button("Load Existing Knowledge Base"):
    with st.spinner("Loading existing vector database..."):
        rag.load_vector_db()
    st.success("Vector DB loaded!")

# Query input
question = st.text_input("Ask a question about carrier appetite:")
llm_choice = st.selectbox("LLM Backend", ["ollama", "openai", "hf"])
top_k = st.slider("Number of retrieved chunks:", min_value=1, max_value=10, value=3)

if question:
    with st.spinner("Retrieving and generating answer..."):
        answer, docs = rag.answer_question(question, llm=llm_choice, top_k=top_k)
    st.write("**Answer:**")
    st.write(answer)

    # Show sources
    st.write("**Source Chunks:**")
    for doc in docs:
        st.write(f"- **{doc.metadata.get('source', '?')}**, page {doc.metadata.get('page', '?')}")
