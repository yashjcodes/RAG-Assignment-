"""OPTIONAL generation step. Runs only if ANTHROPIC_API_KEY is set; otherwise the
pipeline is retrieval-only (stated in README)."""
import os


def generate_answer(question: str, chunks: list[str]):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    ctx = "\n\n---\n\n".join(chunks)
    msg = anthropic.Anthropic().messages.create(
        model="claude-haiku-4-5", max_tokens=400,
        messages=[{"role": "user", "content":
                   f"Answer ONLY from the context. If the answer is not in the context, "
                   f"say 'Not found in the document'.\n\nContext:\n{ctx}\n\nQuestion: {question}"}])
    return msg.content[0].text
