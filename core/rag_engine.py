
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


load_dotenv()


# --------------------------------------------------
# 1. GET LLM
# --------------------------------------------------

def get_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    return ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_tokens=2000,
        max_retries=3,
        timeout=120
    )


# --------------------------------------------------
# 2. FORMAT DOCUMENTS
# --------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# --------------------------------------------------
# 3. CREATE RAG CHAIN
# --------------------------------------------------

def build_rag_chain(transcript: str):

    print("Building RAG chain...")

    # Create vector store
    vector_store = build_vector_store(transcript)

    # Create retriever
    retriever = get_retriever(vector_store, k=4)

    # Load LLM
    llm = get_llm()

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    # Create RAG chain
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain created successfully!")

    return rag_chain


# --------------------------------------------------
# 4. LOAD EXISTING RAG CHAIN
# --------------------------------------------------

def load_rag_chain():

    print("Loading existing RAG chain...")

    # Load existing vector store
    vector_store = load_vector_store()

    # Create retriever
    retriever = get_retriever(vector_store, k=4)

    # Load LLM
    llm = get_llm()

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    # Create RAG chain
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Existing RAG chain loaded successfully!")

    return rag_chain


# --------------------------------------------------
# 5. ASK QUESTION
# --------------------------------------------------

def ask_question(rag_chain, question: str) -> str:

    print(f"\nQuestion: {question}")

    answer = rag_chain.invoke(question)

    print(f"\nAnswer: {answer}")

    return answer
