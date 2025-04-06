from pydub import AudioSegment
import time


def stream_audio(audio: AudioSegment, frame_ms=200):
    """
    Simulates real-time audio streaming.

    Yields small audio chunks of `frame_ms` milliseconds.

    Example: 200ms, like real mic input.
    """

    for i in range(0, len(audio), frame_ms):
        chunk = audio[i:i + frame_ms]
        yield chunk
        time.sleep(frame_ms / 1000.0)
