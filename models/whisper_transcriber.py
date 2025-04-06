import whisper
import numpy as np
import time


class WhisperTranscriber:
    def __init__(self, model_size="base.en", sample_rate=16000):
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")
        self.buffer = []
        self.sample_rate = sample_rate


    def add_audio_chunk(self, audio_chunk):
        """
        Adds an audio chunk to the buffer.

        Parameters:
            audio_chunk (AudioSegment): Audio chunk to be added.
        Returns:
            None
        """
        samples = np.array(audio_chunk.get_array_of_samples()).astype(np.float32) / 32768.0
        self.buffer.extend(samples)


    def transcribe_audio(self, min_chunk_duration_ms=2000, max_chunk_duration_ms=5000):

        """ 
        Transcribes the audio in the buffer. If there's not enough data, it waits until more data is available. 
        (depending on the model size, it may take a while to transcribe) 
        and min_chunk_duration_ms determines minimum amount of data in buffer required to start transcribing.
        
        Parameters:
            min_chunk_duration_ms (int): Minimum duration of audio chunk to start transcribing (in milliseconds).
            max_chunk_duration_ms (int): Maximum duration of audio chunk to transcribe (in milliseconds).
        
        Returns:    
            str: Transcribed text.
        """

        min_samples = int(self.sample_rate * (min_chunk_duration_ms / 1000))
        max_samples = int(self.sample_rate * (max_chunk_duration_ms / 1000))

        if len(self.buffer) < min_samples:
            return None

        chunk_samples = self.buffer[:max_samples]
        self.buffer = self.buffer[max_samples:]

        result = self.model.transcribe(np.array(chunk_samples))
        return result["text"].strip()

