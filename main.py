
from dotenv import load_dotenv

from utils.audio_processors import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


# --------------------------------------------------
# 1. RUN COMPLETE VIDEO PROCESSING PIPELINE
# --------------------------------------------------

def run_pipeline(source: str, language: str = "english") -> dict:

    print("\nStarting AI Video Assistant...")
    print("=" * 60)

    # Step 1: Extract audio and create chunks
    chunks = process_input(source)

    print(f"\nTotal audio chunks: {len(chunks)}")

    # Step 2: Transcribe audio
    print("\nTranscribing audio...")

    transcript = transcribe_all(
        chunks,
        language=language
    )

    print("\nTranscription completed!")
    print("\nRaw transcription (first 300 characters):")
    print(transcript[:300])

    # Step 3: Generate video title
    print("\nGenerating video title...")

    title = generate_title(transcript)

    # Step 4: Generate video summary
    print("\nGenerating video summary...")

    summary = summarize(transcript)

    # Step 5: Extract action items
    print("\nExtracting action items...")

    action_items = extract_action_items(transcript)

    # Step 6: Extract key decisions
    print("\nExtracting key decisions...")

    decisions = extract_key_decisions(transcript)

    # Step 7: Extract open questions
    print("\nExtracting open questions...")

    questions = extract_questions(transcript)

    # Step 8: Build RAG chain
    print("\nBuilding RAG system...")

    rag_chain = build_rag_chain(transcript)

    print("\nVideo processing completed successfully!")

    # Return all results
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain
    }


# --------------------------------------------------
# 2. DISPLAY VIDEO RESULTS
# --------------------------------------------------

def display_results(result: dict):

    print("\n" + "=" * 60)
    print("VIDEO SUMMARIZER RESULTS")
    print("=" * 60)

    print(f"\nTitle:\n{result['title']}")

    print(f"\nSummary:\n{result['summary']}")

    print(f"\nAction Items:\n{result['action_items']}")

    print(f"\nKey Decisions:\n{result['key_decisions']}")

    print(f"\nOpen Questions:\n{result['open_questions']}")

    print("\n" + "=" * 60)


# --------------------------------------------------
# 3. CHAT WITH YOUR VIDEO USING RAG
# --------------------------------------------------

def chat_with_video(rag_chain):

    print("\nChat with your video!")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:

        question = input("You: ").strip()

        # Exit condition
        if question.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        # Skip empty input
        if not question:
            continue

        # Get answer from RAG
        try:
            answer = ask_question(
                rag_chain,
                question
            )

            print(f"\nAssistant: {answer}\n")

        except Exception as e:
            print(f"\nError generating answer: {e}\n")


# --------------------------------------------------
# 4. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AI VIDEO SUMMARIZER AND MEETING ASSISTANT")
    print("=" * 60)

    source = input(
        "\nEnter YouTube URL or audio/video file path: "
    ).strip()

    language = input(
        "Enter video language (english/hindi): "
    ).strip().lower()

    if language not in ["english", "hindi"]:
        print("Invalid language! Please choose english or hindi.")
        return

    try:

        # Run complete pipeline
        result = run_pipeline(
            source,
            language
        )

        # Display generated results
        display_results(result)

        # Start RAG chat
        rag_chain = result["rag_chain"]

        chat_with_video(rag_chain)

    except Exception as e:

        print(f"\nAn error occurred: {e}")


# --------------------------------------------------
# 5. PROGRAM EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    main()
