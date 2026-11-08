
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()


# Initialize OpenRouter LLM

def get_llm():
    return ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_tokens=2000,
        max_retries=3,
        timeout=120
    )



# Split transcript into smaller chunks
def split_transcript(transcript: str) -> list:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


# Summarize transcript
def summarize(transcript: str) -> str:

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}")
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    # Split transcript
    chunks = split_transcript(transcript)

    # Summarize each chunk
    summaries = []

    for i, chunk in enumerate(chunks):

        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")

        summary = map_chain.invoke({"text": chunk})

        summaries.append(summary)

    # Combine individual summaries
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Combine the following summaries into one clear, well-organized and concise summary. Remove repeated information and retain important points."),
            ("human", "{text}")
        ]
    )

    final_chain = final_prompt | llm | StrOutputParser()

    final_summary = final_chain.invoke(
        {"text": "\n\n".join(summaries)}
    )

    return final_summary

# Generate a short meeting title
def generate_title(transcript: str) -> str:

    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else."
            ),
            ("human", "{text}")
        ])
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])
