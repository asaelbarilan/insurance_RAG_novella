import os
import unittest

from rag_pipeline import RAGPipeline

# "Gold standard" answers in the form:
#    question_string: [list_of_expected_carriers]
# taken with gpt O1

gold_answers = {
    "Which carriers provide coverage in Arizona?": [
        "Lynx", "Moxie", "Semsee", "Convex", "Denali"
    ],
    "Which carrier can write a premium of 5,000 USD or lower ?": [
        # $2,500 min => Moxie
        # $5,000 min => Lynx
        "Moxie", "Lynx"
    ],
    "Which carrier can handle a business of type Vacant?": [
        # Lynx, Semsee, Convex, Denali each mention "Vacant"
        "Lynx", "Semsee", "Convex", "Denali"
    ],
    "Find carriers that write apartment buildings with limits over $10M": [
        "Lynx", "Moxie", "Convex"
    ],
    "Which carriers would consider a $20M TIV warehouse in Arizona?": [
        # Convex is the best fit for full coverage
        # Lynx can do partial up to $15M ground-up, or $10M primary if layering
        "Convex", "Lynx"
    ]
}


class TestRAGValidationSet(unittest.TestCase):
    def setUp(self):
        """
        1. Reads the environment variable LLM_CHOICE (optional).
        2. Instantiates a RAGPipeline pointing to the JSON data.
        3. Creates or loads the Chroma DB.
        """
        # Read environment variable or default to "ollama"
        self.llm_choice = os.environ.get("LLM_CHOICE", "ollama")
        print(f"[INFO] Using LLM backend: {self.llm_choice}")

        # Initialize pipeline with JSON-based ingestion
        self.pipeline = RAGPipeline(
            json_path="./data/guide_novella.json",  # Adjust path if needed
            vector_db_path="./chroma_db",
            openai_api_key=None
        )

        # If there's no existing Chroma index, create it
        if not os.path.exists("./chroma_db/index"):
            print("[INFO] Creating fresh vector DB...")
            self.pipeline.create_vector_db()
        else:
            print("[INFO] Loading existing vector DB...")
            self.pipeline.load_vector_db()

    def test_validation_set(self):
        """
        Runs each question in 'gold_answers' against the pipeline,
        then checks if the LLM response contains all expected carriers.
        """
        for question, expected_carriers in gold_answers.items():
            print(f"\n[TEST] Question: {question}")
            answer, docs = self.pipeline.answer_question(
                question=question,
                llm=self.llm_choice,
                top_k=5
            )
            print(f"LLM Answer:\n{answer}\n")

            # Basic check: we got some answer
            self.assertIsNotNone(answer, "LLM returned no answer.")

            # Simple string matching for each expected carrier name
            normalized_answer = answer.lower()
            for carrier in expected_carriers:
                carrier_lower = carrier.lower()
                self.assertIn(
                    carrier_lower,
                    normalized_answer,
                    msg=f"Expected '{carrier}' to appear in the answer for '{question}'"
                )


if __name__ == "__main__":
    unittest.main()
