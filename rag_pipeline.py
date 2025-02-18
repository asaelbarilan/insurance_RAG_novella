import os
import json
import requests
import openai

from typing import List, Optional
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def chunk_documents_with_metadata(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Takes a list of LangChain Document objects (docs),
    and returns a new list of Documents split by chunk_size,
    each retaining the metadata from the original doc.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc_index, doc in enumerate(docs):
        if not isinstance(doc, Document):
            print(f"[WARN] doc #{doc_index} is not a Document. It's {type(doc)}.")
            continue

        original_metadata = doc.metadata.copy()
        page_text = doc.page_content
        chunks = text_splitter.split_text(page_text)

        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={**original_metadata, "chunk_index": i}
            )
            all_chunks.append(chunk_doc)

    print(f"[INFO] Produced {len(all_chunks)} total text chunks.")
    return all_chunks


class RAGPipeline:
    """
    A pipeline that:
      1. Ingests data from a JSON file, where each entry has { carrier_name, guide }.
      2. Splits them into chunks for efficient retrieval.
      3. Stores them in a Chroma vector database.
      4. Uses LLM (local or OpenAI) to generate final answers.
    """

    def __init__(
        self,
        json_path: str,
        vector_db_path: str,
        openai_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        """
        :param json_path: Path to the JSON file containing the guides.
        :param vector_db_path: Directory to create/load the Chroma DB.
        :param openai_api_key: Your OpenAI API key (if using OpenAI).
        :param ollama_url: Endpoint for local Ollama server.
        """
        self.json_path = json_path
        self.vector_db_path = vector_db_path
        self.ollama_url = ollama_url

        # Optionally set your OpenAI API key
        if openai_api_key:
            openai.api_key = openai_api_key

        # Use HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_db = None

    def load_json_as_documents(self) -> List[Document]:
        """
        Reads the JSON file, converts each carrier's 'guide' text into
        a LangChain Document, and attaches metadata (carrier_name).
        Structure example:
        [
          {
            "carrier_name": "Lynx",
            "guide": "...some multiline text..."
          },
          ...
        ]
        """
        if not os.path.exists(self.json_path):
            print(f"[ERROR] JSON file not found: {self.json_path}")
            return []

        print(f"[INFO] Loading JSON data from {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for entry in data:
            carrier_name = entry.get("carrier_name", "UnknownCarrier")
            guide_text = entry.get("guide", "")
            if not guide_text.strip():
                # skip empty text
                continue
            # Create a Document for each carrier's guide
            doc = Document(
                page_content=guide_text,
                metadata={"carrier_name": carrier_name}
            )
            documents.append(doc)

        print(f"[INFO] Loaded {len(documents)} Documents from JSON.")
        return documents

    def load_and_chunk_docs(self) -> List[Document]:
        """
        1. Load docs from JSON
        2. Chunk them into smaller pieces
        """
        docs = self.load_json_as_documents()
        # Now chunk them
        chunked_docs = chunk_documents_with_metadata(docs)
        return chunked_docs

    def create_vector_db(self):
        """
        Creates (and persists) a new Chroma vector store from the loaded chunks.
        """
        print("[INFO] Creating vector DB from JSON documents...")
        chunks = self.load_and_chunk_docs()
        if not chunks:
            print("[WARN] No chunks found. Check your JSON or chunking.")
            return

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        self.vector_db.persist()
        print(f"[INFO] Vector database created & persisted at '{self.vector_db_path}'.")

    def load_vector_db(self):
        """
        Loads an existing Chroma vector store from disk.
        """
        if not os.path.exists(self.vector_db_path):
            print(f"[ERROR] Vector DB path '{self.vector_db_path}' not found.")
            return

        print(f"[INFO] Loading existing Chroma DB from '{self.vector_db_path}'.")
        self.vector_db = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )

    def ask_ollama(self, prompt: str, model_name: str = "llama3") -> str:
        """
        Query the local Ollama server for Llama-based inference.
        Adjust the model_name to match what's installed in Ollama.
        """
        print(f"[DEBUG] Sending prompt to Ollama with model={model_name}")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.ollama_url, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise RuntimeError(f"[ERROR] Ollama call failed: {response.status_code} {response.text}")

    def ask_openai(self, prompt: str, model_name: str = "gpt-3.5-turbo") -> str:
        """
        Query OpenAI ChatCompletion (GPT-3.5/4).
        """
        print(f"[DEBUG] Sending prompt to OpenAI with model={model_name}")
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response["choices"][0]["message"]["content"]

    def ask_huggingface(self, prompt: str) -> str:
        """
        Placeholder for a local HuggingFace pipeline.
        """
        return "[DEBUG] HuggingFace local pipeline is not implemented in this example."

    def ask_llm(self, prompt: str, llm: str = "ollama") -> str:
        """
        Dispatch the prompt to your chosen LLM backend: 'ollama', 'openai', or 'hf'.
        """
        if llm == "ollama":
            return self.ask_ollama(prompt)
        elif llm == "openai":
            return self.ask_openai(prompt)
        elif llm == "hf":
            return self.ask_huggingface(prompt)
        else:
            raise ValueError(f"[ERROR] Unknown LLM type: {llm}")

    def answer_question(self, question: str, llm: str = "ollama", top_k: int = 3):
        """
        Retrieve top-k docs from the vector DB, build a prompt, and get an LLM-based answer.

        :return: (answer string, List of retrieved Documents)
        """
        if not self.vector_db:
            print("[INFO] Vector DB not loaded. Attempting to load from disk...")
            self.load_vector_db()
            if not self.vector_db:
                return "[ERROR] No vector DB available.", []

        # Retrieve relevant docs
        retriever = self.vector_db.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.get_relevant_documents(question)
        print(f"[INFO] Retrieved {len(relevant_docs)} relevant chunks.")

        # Build context block
        context_blocks = []
        for doc in relevant_docs:
            carrier = doc.metadata.get("carrier_name", "UnknownCarrier")
            snippet = (
                f"Carrier: {carrier}\n"
                f"{doc.page_content}\n"
                "---------\n"
            )
            context_blocks.append(snippet)

        combined_context = "\n".join(context_blocks)

        # Final prompt
        prompt = f"""You are a helpful assistant. Please provide your answer in clear, coherent sentences. 
        Ensure proper spacing, punctuation, and Markdown formatting.

        Context:
        {combined_context}

        Question: {question}

        Answer:
        """

        answer = self.ask_llm(prompt, llm=llm).strip()

        return answer, relevant_docs


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # 1. Initialize the pipeline
    pipeline = RAGPipeline(
        json_path="./data/guide_novella.json",   # adjust as needed
        vector_db_path="./chroma_db",
        openai_api_key=None
    )

    # 2. Create or load the vector database
    if not os.path.exists("./chroma_db/index"):
        pipeline.create_vector_db()
    else:
        pipeline.load_vector_db()

    # 3. Ask a sample question
    question = "Which carriers provide coverage in Arizona?"
    ans, docs = pipeline.answer_question(question, llm="ollama")
    print("=== LLM ANSWER ===")
    print(ans)
    print("=== RELEVANT DOCS ===")
    for d in docs:
        print(f"- {d.metadata.get('carrier_name')}")
