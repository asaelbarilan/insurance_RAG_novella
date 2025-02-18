# import streamlit as st
# from rag_pipeline import RAGPipeline
#
# st.title("Insurance Carrier RAG System")
# st.write("Use this interface to answer questions about carrier appetites.")
#
# # Initialize our pipeline
# rag = RAGPipeline(
#     json_path="./data/guide_novella.json",
#     vector_db_path="./chroma_db",
#     openai_api_key=None  # or your actual key if using OpenAI
# )
#
# # Buttons to create/load the vector DB
# if st.button("Initialize Knowledge Base"):
#     with st.spinner("Processing PDFs and creating vector database..."):
#         rag.create_vector_db()
#     st.success("Vector DB initialized!")
#
# if st.button("Load Existing Knowledge Base"):
#     with st.spinner("Loading existing vector database..."):
#         rag.load_vector_db()
#     st.success("Vector DB loaded!")
#
# # Query input
# question = st.text_input("Ask a question about carrier appetite:")
# llm_choice = st.selectbox("LLM Backend", ["ollama", "openai", "hf"])
# top_k = st.slider("Number of retrieved chunks:", min_value=1, max_value=10, value=3)
#
# if question:
#     with st.spinner("Retrieving and generating answer..."):
#         answer, docs = rag.answer_question(question, llm=llm_choice, top_k=top_k)
#     st.write("**Answer:**")
#     st.write(answer)
#
#     # Show sources
#     st.write("**Source Chunks:**")
#     for doc in docs:
#         st.write(f"- **{doc.metadata.get('source', '?')}**, page {doc.metadata.get('page', '?')}")

import streamlit as st
import openai
# from anthropic import Anthropic # if you install the anthropic package
# import anthropic

from rag_pipeline import RAGPipeline

st.title("Insurance Carrier RAG System")
st.write("Use this interface to answer questions about carrier appetites.")

# 1. UI for user to pick LLM backend
llm_backend = st.selectbox("LLM Backend", ["ollama", "openai"]) #, "hf", "anthropic-claude"

# 2. Text fields for user to enter keys if needed
openai_api_key = st.text_input("OpenAI API Key (if using OpenAI)", type="password")
#anthropic_api_key = st.text_input("Anthropic API Key (if using Claude)", type="password")
ollama_model_name = st.text_input("Ollama Model Name (if using Ollama)", value="llama2")

# 3. Create the pipeline
#    (only once user clicks a button, so we don't keep re-creating it each time)
if "rag" not in st.session_state:
    st.session_state["rag"] = None  # placeholder

if st.button("Initialize Knowledge Base"):
    # We'll pass openai_api_key if the user chose openai.
    # For Anthropic or other custom LLM, you might need to store the key differently.
    if llm_backend == "openai" and openai_api_key:
        openai.api_key = openai_api_key  # set globally, or pass to the pipeline’s constructor

    st.session_state["rag"] = RAGPipeline(
        json_path="./data/guide_novella.json",
        vector_db_path="./chroma_db",
        openai_api_key=openai_api_key if (llm_backend == "openai") else None,
        ollama_url="http://localhost:11434/api/generate"  # or your local Ollama endpoint
    )

    # Create or load DB
    if not st.session_state["rag"].vector_db:
        st.session_state["rag"].create_vector_db()
    st.success("Knowledge base initialized!")

# Optional: load existing DB (if you already built embeddings in a previous session)
if st.button("Load Existing Knowledge Base"):
    if "rag" in st.session_state and st.session_state["rag"] is not None:
        st.session_state["rag"].load_vector_db()
        st.success("Loaded existing vector DB!")
    else:
        st.warning("RAG pipeline not initialized yet.")

# 4. Query input
question = st.text_input("Ask a question about carrier appetite:")
top_k = st.slider("Number of retrieved chunks:", min_value=1, max_value=10, value=6)

# 5. Ask the question
if question and st.session_state["rag"] is not None:
    with st.spinner("Retrieving and generating answer..."):
        # We pass the chosen backend in the 'llm' arg
        answer, docs = st.session_state["rag"].answer_question(
            question,
            llm=llm_backend,
            top_k=top_k
        )
    st.write("**Answer:**")
    st.write(answer)

    # Show sources
    st.write("**Source Chunks:**")
    for doc in docs:
        # For JSON-based docs, you might show 'carrier_name' or chunk_index
        carrier_name = doc.metadata.get("carrier_name", "?")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        st.write(f"- {carrier_name}, chunk {chunk_idx}")
else:
    st.write("Enter a question and/or initialize the pipeline above.")
