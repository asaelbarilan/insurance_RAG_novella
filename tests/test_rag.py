import os
import shutil
import unittest
from rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        # Prepare a test folder, e.g. "test_data"
        self.test_pdf_folder = "./test_data"
        self.test_db_path = "./test_chroma_db"
        os.makedirs(self.test_pdf_folder, exist_ok=True)

        # You can place small dummy PDFs here or mock them.
        # For now, assume there's a sample PDF in ./test_data.

    def tearDown(self):
        # Clean up
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        if os.path.exists(self.test_pdf_folder):
            shutil.rmtree(self.test_pdf_folder)

    def test_create_vector_db(self):
        pipeline = RAGPipeline(
            pdf_folder=self.test_pdf_folder,
            vector_db_path=self.test_db_path
        )
        # This should handle empty or real PDFs
        pipeline.create_vector_db()
        self.assertTrue(os.path.exists(self.test_db_path))

    def test_answer_question(self):
        pipeline = RAGPipeline(
            pdf_folder=self.test_pdf_folder,
            vector_db_path=self.test_db_path
        )
        # Create a fresh vector DB (or you can pre-load if you already have one)
        pipeline.create_vector_db()

        # 1. Which carriers provide coverage in Arizona?
        question1 = "Which carriers provide coverage in Arizona?"
        answer1, docs1 = pipeline.answer_question(question1, llm="hf")
        self.assertIsNotNone(answer1)
        print(f"\nQ1: {question1}\nAnswer: {answer1}\n")

        # 2. Which carrier can write a premium of X USD or lower?
        question2 = "Which carriers can write a premium of 5,000 USD or lower?"
        answer2, docs2 = pipeline.answer_question(question2, llm="hf")
        self.assertIsNotNone(answer2)
        print(f"\nQ2: {question2}\nAnswer: {answer2}\n")

        # 3. Which carrier can handle a business of type X?
        # Example: "Which carrier can handle a vacant property?"
        question3 = "Which carriers can handle vacant properties?"
        answer3, docs3 = pipeline.answer_question(question3, llm="hf")
        self.assertIsNotNone(answer3)
        print(f"\nQ3: {question3}\nAnswer: {answer3}\n")

        # 4. Find carriers that write apartment buildings with limits over $10M
        question4 = "Find carriers that write apartment buildings with limits over $10M."
        answer4, docs4 = pipeline.answer_question(question4, llm="hf")
        self.assertIsNotNone(answer4)
        print(f"\nQ4: {question4}\nAnswer: {answer4}\n")

        # 5. Which carriers would consider a $20M TIV warehouse in Arizona?
        question5 = "Which carriers would consider a $20M TIV warehouse in Arizona?"
        answer5, docs5 = pipeline.answer_question(question5, llm="hf")
        self.assertIsNotNone(answer5)
        print(f"\nQ5: {question5}\nAnswer: {answer5}\n")


if __name__ == "__main__":
    unittest.main()
