import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REPO_ID = "llama-3.1-8b-instant"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_llm():
    return ChatGroq(
        model=REPO_ID,
        temperature=0.5,
        max_retries=2,
    )


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know. Do not make up information.

Context:
{context}

Question:
{question}

Start the answer directly without small talk.
"""


def get_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


def main():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | get_prompt()
        | load_llm()
        | StrOutputParser()
    )

    user_query = input("Write query here: ")

    result = rag_chain.invoke(user_query)
    source_docs = retriever.get_relevant_documents(user_query)

    print("\nRESULT:\n", result)
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(source_docs, 1):
        metadata = doc.metadata
        page = metadata.get("page_label", metadata.get("page", "Unknown"))
        source = metadata.get("source", "Unknown").split("\\")[-1]
        print(f"{i}. {source} (Page {page})")


if __name__ == "__main__":
    main()
