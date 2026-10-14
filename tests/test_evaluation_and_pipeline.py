import sys
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import judge
from generate import generate_answer
from questions import QUESTIONS


class TestEvaluationAndPipeline(unittest.TestCase):
    def test_judge_yes(self):
        # Q1 requires 6.5% and 126 mg/dl
        q1 = QUESTIONS[0]
        retrieved_text = "Diagnosis requires HbA1c >= 6.5% and fasting glucose >= 126 mg/dL."
        verdict = judge(q1, retrieved_text)
        self.assertEqual(verdict, "yes")

    def test_judge_partial(self):
        # Q1 with only one fact satisfied
        q1 = QUESTIONS[0]
        retrieved_text = "Fasting plasma glucose was measured at 126 mg/dL."
        verdict = judge(q1, retrieved_text)
        self.assertEqual(verdict, "partial")

    def test_judge_no(self):
        # Q1 with no relevant facts
        q1 = QUESTIONS[0]
        retrieved_text = "The patient was advised to exercise regularly."
        verdict = judge(q1, retrieved_text)
        self.assertEqual(verdict, "no")

    def test_judge_unanswerable(self):
        # Q9 is unanswerable from the document
        q9 = QUESTIONS[8]
        self.assertFalse(q9["answerable"])
        verdict = judge(q9, "Some arbitrary text about glucose")
        self.assertEqual(verdict, "n/a")

    def test_generate_answer_no_api_key(self):
        # Should safely return None when ANTHROPIC_API_KEY is unset
        import os
        old_val = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            res = generate_answer("What is diabetes?", ["Sample chunk text"])
            self.assertIsNone(res)
        finally:
            if old_val is not None:
                os.environ["ANTHROPIC_API_KEY"] = old_val


if __name__ == "__main__":
    unittest.main()
