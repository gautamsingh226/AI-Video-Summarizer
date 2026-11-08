
import os
import yt_dlp
from pydub import AudioSegment


# ---------------------------------
# 1. DOWNLOAD DIRECTORY
# ---------------------------------

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ---------------------------------
# 2. DOWNLOAD YOUTUBE AUDIO
# ---------------------------------

def download_youtube_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Downloaded audio not found: {filename}"
            )

        print("Audio downloaded successfully!")
        print("Audio path:", filename)

        return filename

    except Exception as e:
        print("Download error:", str(e))
        raise


# ---------------------------------
# 3. CONVERT AUDIO TO WAV
# ---------------------------------

def convert_to_wav(input_path: str) -> str:

    """Convert any audio/video file to WAV format using pydub."""

    output_path = (
        os.path.splitext(input_path)[0] + "_converted.wav"
    )

    audio = AudioSegment.from_file(input_path)

    # Convert to mono and 16 kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    print("Audio converted successfully!")
    print("Converted file:", output_path)

    return output_path


# ---------------------------------
# 4. CHUNKING THE AUDIO FILE
# ---------------------------------

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start : start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


# ---------------------------------
# 5. PROCESS INPUT
# ---------------------------------

def process_input(source: str) -> list:

    if source.startswith("http://") or source.startswith("https://"):

        print("Detected YouTube URL. Downloading audio...")

        wav_path = download_youtube_audio(source)

        wav_path = convert_to_wav(wav_path)

    else:

        print("Detected local file. Converting to WAV...")

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(f"Audio ready - {len(chunks)} chunk(s) created.")

    return chunks


# ---------------------------------
# 6. MAIN EXECUTION
# ---------------------------------

if __name__ == "__main__":

    video_url = "https://www.youtube.com/watch?v=nJNsOLJ0rTo&t=360s"

    chunks = process_input(video_url)

    print("\nGenerated audio chunks:")

    for chunk in chunks:
        print(chunk)
