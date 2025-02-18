# Insurance  RAG System

This repository contains a Retrieval-Augmented Generation (RAG) system to help insurance brokers quickly 
answer questions about carriers’ appetites. It ingests multiple carrier appetite guides in PDF format, 
stores embeddings in a vector database, and uses an LLM to generate answers based on retrieved content.

## Features

1. **Document Processing**
   - Loads PDFs of carrier appetites from `./data`.
   - Splits them into manageable chunks and stores them in a vector database (Chroma in this example).

2. **Query System**
   - A simple Python-based interface (either Streamlit or CLI).
   - Supports local LLM inference (via Ollama for Llama models) or OpenAI (GPT-3.5/4).
   - Accurate retrieval of relevant information from the vector store.

3. **Example Questions**
   - "Which carriers provide coverage in Arizona?"
   - "Which carrier can write a premium of 50,000 USD or lower?"
   - "Find carriers that write apartment buildings with limits over \$10M."
   - "Which carriers would consider a \$20M TIV warehouse in Arizona?"
   - etc.

## Setup Instructions

1. **Clone the repo**:

   ```bash
   git clone https://github.com/asaelbarilan/insurance_RAG_novella.git
   cd insurance_RAG_novella

2.activate:
	-put guide_novella.json in data folder
	-run app UI in cmd (from folder): streamlit run app.py
	-run tests: python -m unittest tests/test_rag.py

**notes:**
	-Initialize (create DB) vs. Load (use existing DB) is about whether you’re building new embeddings or re-using what’s 		already built.
	-The chunk slider during queries is about how many results come back from the existing database—not the original text-	splitting chunk size.
