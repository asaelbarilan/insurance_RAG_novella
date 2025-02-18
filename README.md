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

2. **activate**:
   - put guide_novella.json in data folder
   - run app UI in cmd (from folder): streamlit run app.py
	- if using open ai put key inside UI ( its single use and wont be saved)
	- to run local model I used ollama with llama 3.1 8b params, the model need to be downloaded first
   - run tests: python -m unittest tests/test_rag.py

3. **notes:**
   - Initialize (create DB) vs. Load (use existing DB) is about whether you’re building new embeddings or re-using what’s already built.
   - The chunk slider during queries is about how many results come back from the existing database—not the original text-splitting chunk size.

## Design Decisions

### Chroma as Vector Store
- We use Chroma for persistent storage of embeddings.
- Chroma is open-source and integrates easily with LangChain.

### Embeddings
- By default, the system uses a HuggingFace model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- This model is lightweight and offers reasonable performance for short texts.

### LLM Backends
- **Local** via Ollama for Llama-based models for saving money on unnecessary calls. 
- **OpenAI** (GPT-3.5/GPT-4) if you provide an API key.
- **Optional placeholders** for Hugging Face local pipelines, Anthropic (Claude), etc. for future use

- **Chunking & Retrieval** Fine-tuned chunk sizes and adjusted “top_k” retrieval to ensure the relevant carriers aren’t missed.
- **Prompt Engineering** Improved prompt instructions to list all matching carriers, not just the first.
- **chunks** chunks are in json per carrier
- **testing** created the answers to relevant questions with O1 

---

## Known Limitations + future work


- **Hallucination Risk**  
  The LLM can generate text not supported by any retrieved source- add second prompt for verification 

- **Lack of Full Structured Metadata**  
  meta data and normalization may help. storing explicit metadata fields. 


- **Model & Rate Limits**  
  If using OpenAI, you can hit rate/usage limits; local models require adequate hardware resources.
but small local models can be "less smart".

 
- **Adopt Chain-of-Thought**
Break down the LLM’s reasoning steps (e.g., parse coverage states first, then generate the final answer).  

- **tests**
Broaden your set of known Q&A pairs

- **one shot learning**
add one shot to prompts so the model would understand better how to search for the question.


