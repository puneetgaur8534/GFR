"""
Core RAG logic. Imported by app.py (Stack Board UI) and evaluate.py (RAGAS testing).
Do not run this file directly for normal use — run app.py instead.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

PROMPT_TEMPLATE = """You are an assistant answering questions about the General Financial Rules (GFR) document.
Use ONLY the context below to answer the question.
If the answer is not present in the context, say "I don't know based on the given document."
Do not make up information.

Context:
{context}

Question:
{question}

Answer:"""


def get_vectorstore():
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTORSTORE_DIR,
        embedding_function=embeddings,
    )
    return vectordb


def get_retriever(k: int = None):
    if k is None:
        k = config.TOP_K
    vectordb = get_vectorstore()
    return vectordb.as_retriever(search_kwargs={"k": k})


def get_llm():
    return ChatOllama(model=config.LLM_MODEL, temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_query(question: str, k: int = None) -> dict:
    """
    Runs retrieval + generation for one question.
    Returns a dict with question, answer, and the retrieved context chunks
    (contexts is a list of strings — the format RAGAS expects).
    """
    retriever = get_retriever(k=k)
    llm = get_llm()

    docs = retriever.invoke(question)
    context_text = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context_text, "question": question})

    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in docs],
    }


if __name__ == "__main__":
    # quick manual sanity check
    result = answer_query("What is GFR?")
    print("Q:", result["question"])
    print("A:", result["answer"])
    print("Retrieved chunks:", len(result["contexts"]))