import numpy as np
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel
from multiprocessing import Process, Queue

class WhisperTranscriber:
    def __init__(self, model_type = "base.en", language = None,  device = "cpu",
                 sample_rate = 16000, min_chunk_duration_ms = 4000, max_chunk_duration_ms = 5000, 
                 vad_threshold = 0.65, vad_min_silence_ms = 150, vad_pad_ms = 150, silence_length = 12, overlap_length = 0.1):

        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_chunk_duration_ms / 1000)
        self.max_samples = int(sample_rate * max_chunk_duration_ms / 1000)

        # Loading STT model
        print(f"Loading Faster Whisper model: {model_type} …")
        self.model = WhisperModel(model_type, compute_type="auto", device=device)
        print("Whisper model loaded!")

        # Loading VAD model
        print("Loading Silero-VAD …")
        vad_model = load_silero_vad()
        self.vad = VADIterator(vad_model, threshold=vad_threshold, min_silence_duration_ms=vad_min_silence_ms, speech_pad_ms=vad_pad_ms)
        print("Silero-VAD initialised!")

        self.speech_buffer = [] # Stores audio as it comes in
        self.transcript_context = []
        self.vad_residual = [] # Stores leftover audio samples
        self.silence_counter = 0 # Keeps track of consecutive silence frames
        self.silence_length = silence_length # Parameter to determine enough silence for chunking
        self.overlap = int(sample_rate * overlap_length)

    def add_audio_chunk(self, audio_segment):
        samples = (np.array(audio_segment.get_array_of_samples())/32768.0)

        self.vad_residual.extend(samples)

        frame_size = 512
        transcript_out = None

        while len(self.vad_residual) >= frame_size:
            frame = np.array(self.vad_residual[:frame_size])
            del self.vad_residual[:frame_size]

            is_speech = self.vad(frame, return_seconds=False)

            if is_speech:
                self.speech_buffer.extend(frame)
                self.silence_counter = 0
            else:
                self.silence_counter += 1
                self.speech_buffer.extend(frame)

            long_enough = len(self.speech_buffer) >= self.min_samples
            enough_silence = self.silence_counter >= self.silence_length

            if (enough_silence and long_enough) or len(self.speech_buffer) >= self.max_samples: # Enough silence or max chunk size reached
                print(int(len(self.speech_buffer))/int(self.sample_rate))

                keep_for_next = self.speech_buffer[-self.overlap:]
                chunk = np.array(self.speech_buffer[:-self.overlap])

                self.speech_buffer = list(keep_for_next)
                self.silence_counter = 0
                transcript_out = self.transcribe_buffer(chunk)
                break


        return transcript_out

    def transcribe_buffer(self,chunk):
        if len(chunk) == 0:
            return None
        segments, _ = self.model.transcribe(chunk, initial_prompt=" ".join(self.transcript_context))
        text = " ".join(s.text.strip() for s in segments)
        if text:
            self.transcript_context.append(text)
        return text
    
    # Should be called at the end of transcription to flush any remaining audio
    def flush(self):
        if self.vad_residual:
            self.speech_buffer.extend(self.vad_residual)
            self.vad_residual.clear()

        if self.speech_buffer:
            chunk = np.array(self.speech_buffer)
            self.speech_buffer.clear()
            self.silence_counter = 0
            return self.transcribe_buffer(chunk)

        return None