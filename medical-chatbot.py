import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="MedBot - AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db


def set_custom_template(custom_prompt_template: str) -> PromptTemplate:
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


def load_llm(repo_id: str):
    return ChatGroq(
        model=repo_id,
        temperature=0.5,
        max_retries=2,
    )


def main():
    st.title("🏥 MedBot - AI Medical Assistant")

    # Sidebar
    with st.sidebar:
        st.header("📖 How to Use")
        st.write("**Step 1:** Type your medical question in the chat")
        st.write("**Step 2:** Press Enter")
        st.write("**Step 3:** Get AI-powered answers from medical documents")
        st.write("**Step 4:** View sources to verify information")

        st.divider()

        st.header("📐 Architecture Overview")
        st.write("• PDF-based medical knowledge base")
        st.write("• Text chunking + embeddings (MiniLM)")
        st.write("• FAISS similarity search")
        st.write("• Groq LLaMA 3.1 (8B)")
        st.write("• LCEL-based RAG pipeline")
        st.write("• Streamlit frontend")

        st.divider()

        if st.button("🔄 New Conversation", type="primary"):
            st.session_state.messages = []
            st.rerun()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Chat input
    user_prompt = st.chat_input("Ask your medical question...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

        custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know.
Do not make up information.

Context:
{context}

Question:
{question}

Start the answer directly without small talk.
"""

        repo_id = "llama-3.1-8b-instant"

        try:
            with st.spinner("🧠 Analyzing your question..."):
                vectorstore = get_vectorstore()
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                prompt_template = set_custom_template(custom_prompt_template)
                llm = load_llm(repo_id)

                rag_chain = (
                    {
                        "context": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                result = rag_chain.invoke(user_prompt)
                source_docs = retriever.get_relevant_documents(user_prompt)

                # Format sources
                formatted_sources = ""
                if source_docs:
                    formatted_sources = "\n\n**📚 Sources:**\n"
                    for i, doc in enumerate(source_docs, 1):
                        metadata = doc.metadata
                        page = metadata.get(
                            "page_label",
                            metadata.get("page", "Unknown"),
                        )
                        source = (
                            metadata.get("source", "Unknown")
                            .split("\\")[-1]
                            .split("/")[-1]
                        )
                        formatted_sources += (
                            f"**{i}.** {source} (Page {page})\n"
                        )

                final_answer = result + formatted_sources

                st.chat_message("assistant").markdown(final_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )

        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            st.chat_message("assistant").markdown(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

    st.divider()
    st.markdown(
        """
**⚠️ Medical Disclaimer:**  
This AI assistant provides information for educational purposes only.  
Always consult a qualified healthcare professional for medical advice.
"""
    )


if __name__ == "__main__":
    main()
