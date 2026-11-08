
import os
import whisper
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

_model = None

# Sarvam AI client
client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
) if SARVAM_API_KEY else None


# Load Whisper model
def load_model():

    global _model

    if _model is None:
        print("Loading Whisper model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully!")

    return _model


# Transcribe audio using Whisper
def transcribe_chunk(chunk_path: str, language: str) -> str:

    model = load_model()

    print(f"Starting transcription: {chunk_path}")

    result = model.transcribe(
        chunk_path,
        language=language,
        task="transcribe",
        fp16=False,
        beam_size=1,
        best_of=1,
        verbose=False,
        condition_on_previous_text=False
    )

    print("Chunk transcription completed!")

    return result["text"]


# Split long Hindi text into smaller chunks
def split_text(text: str, max_chars: int = 1800) -> list:

    words = text.split()
    chunks = []
    current = ""

    for word in words:

        if len(current) + len(word) + 1 <= max_chars:
            current += word + " "
        else:
            if current.strip():
                chunks.append(current.strip())

            current = word + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks


# Translate Hindi into English using Sarvam AI
def translate_to_english(text: str) -> str:

    if client is None:
        raise ValueError("SARVAM_API_KEY not found in .env")

    text_chunks = split_text(text)
    translated_text = []

    for i, chunk in enumerate(text_chunks):

        print(f"Sarvam translating chunk {i + 1}...")

        response = client.text.translate(
            input=chunk,
            source_language_code="hi-IN",
            target_language_code="en-IN",
            model="sarvam-translate:v1"
        )

        translated_text.append(response.translated_text)

    return " ".join(translated_text)


# Transcribe all audio chunks
def transcribe_all(
    chunks: list,
    language: str = "hi"
) -> str:

    full_transcript = ""

    for i, chunk in enumerate(chunks):

        print(f"Transcribing audio chunk {i + 1}...")

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

    full_transcript = full_transcript.strip()

    print("Transcription completed!")

    # Hindi audio: Translate using Sarvam
    if language.lower() in ["hi", "hindi", "hi-in"]:

        print("Translating Hindi transcript using Sarvam AI...")

        full_transcript = translate_to_english(full_transcript)

        print("Sarvam translation completed!")

    # English audio: Keep Whisper transcript
    elif language.lower() in ["en", "english", "en-us", "en-in"]:

        print("English transcription completed using Whisper!")

    else:
        raise ValueError("Supported languages are Hindi and English.")

    return full_transcript
