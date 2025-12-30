from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.core.llm import generate_answer

def answer_question(question: str):
    store = VectorStore(384)
    retriever = Retriever(store)

    docs = retriever.retrieve(question)
    context = "\n".join([d["text"] for d in docs])

    answer = generate_answer(context, question)

    return {
        "answer": answer,
        "sources": docs
    }
