from utils.audio_preprocessing import preprocess_audio
import whisper
import numpy as np
import time


class WhisperTranscriber:
    def __init__(self, model_size="base.en"):
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")


    def transcribe_audio_in_chunks(self, audio_file, chunk_duration_ms=5000):
        """
        Transcribes an audio file in chunks.

        Parameters:
            audio_file (str): Path to the audio file.
            chunk_duration_ms (int): Chunk size in milliseconds.

        Returns:
            list: List of transcribed text for each chunk.
        """
        
        # Use the utils function to preprocess the audio
        audio = preprocess_audio(audio_file)

        transcriptions = []
        num_chunks = len(audio) // chunk_duration_ms + (1 if len(audio) % chunk_duration_ms != 0 else 0)

        print("\nStarting streaming transcription...\n")

        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

            result = self.model.transcribe(samples)
            transcription = result["text"].strip()
            print(f"Chunk {i // chunk_duration_ms + 1}: {transcription}")
            transcriptions.append(transcription)

            time.sleep(chunk_duration_ms / 1000.0)

        return transcriptions
    
