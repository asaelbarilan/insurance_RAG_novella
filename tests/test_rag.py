import os
import unittest

from rag_pipeline import RAGPipeline

# "Gold standard" answers in the form:
#    question_string: [list_of_expected_carriers]
gold_answers = {
    "Which carriers provide coverage in Arizona?": [
        "Lynx", "Moxie", "Semsee", "Convex", "Denali"
    ],
    "Which carrier can write a premium of 5,000 USD or lower ?": [
        "Moxie", "Lynx"
    ],
    "Which carrier can handle a business of type Vacant?": [
        "Lynx", "Semsee", "Convex", "Denali"
    ],
    "Find carriers that write apartment buildings with limits over $10M": [
        "Lynx", "Moxie", "Convex"
    ],
    "Which carriers would consider a $20M TIV warehouse in Arizona?": [
        "Convex", "Lynx"
    ]
}


class TestRAGValidationSet(unittest.TestCase):
    def setUp(self):
        """
        Sets up the pipeline by constructing an absolute path to guide_novella.json
        and either creating or loading the Chroma DB. Also defines self.llm to avoid errors.
        """
        # Derive project root from the location of this test file
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        json_path = os.path.join(project_root, "data", "guide_novella.json")

        # Debug print to confirm the path
        print(f"[DEBUG] Using JSON path => {json_path}")

        # Instantiate RAGPipeline with an absolute path
        self.pipeline = RAGPipeline(
            json_path=json_path,
            vector_db_path=os.path.join(project_root, "chroma_db")
        )

        # Define the LLM backend for all tests (e.g., "ollama", "openai", or "hf")
        self.llm = "ollama"

        # Create or load DB
        chroma_index_path = os.path.join(project_root, "chroma_db", "index")
        if not os.path.exists(chroma_index_path):
            self.pipeline.create_vector_db()
        else:
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
                llm=self.llm,  # uses the LLM we set in setUp()
                top_k=5        # if your pipeline supports a 'top_k' parameter
            )
            print(f"LLM Answer:\n{answer}\n")

            # Basic check: we got some answer
            self.assertIsNotNone(answer, "LLM returned no answer.")

            # Simple string matching for each expected carrier
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
