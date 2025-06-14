import numpy as np
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel
from transformers import pipeline

class LiveTranscriber:
    def __init__(self, sample_rate = 16000, min_chunk_duration_ms = 4500, 
                 vad_threshold = 0.65, silence_length_ms = 550, overlap_length = 0.0, device = "cpu"):
        max_chunk_duration_ms = min_chunk_duration_ms + 1000
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_chunk_duration_ms / 1000)
        self.max_samples = int(sample_rate * max_chunk_duration_ms / 1000)

        # Loading VAD model
        print("Loading Silero-VAD …")
        vad_model = load_silero_vad()
        self.vad = VADIterator(vad_model, threshold=vad_threshold)
        print("Silero-VAD initialised!")
        self.speech_buffer = [] # Stores audio as it comes in
        self.vad_residual = [] # Stores leftover audio samples
        self.silence_duration_tracker = 0 # Keeps track silence duration in ms
        self.silence_length_ms = silence_length_ms # Parameter to determine enough silence for chunking
        self.overlap = int(sample_rate * overlap_length) 

    def add_audio_chunk(self, audio_segment):
        samples = (np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0)
        self.vad_residual.extend(samples)
        frame_size = 512
        transcript_out = None
        frame_duration_ms = frame_size * 1000 / self.sample_rate

        while len(self.vad_residual) >= frame_size:
            frame = np.array(self.vad_residual[:frame_size])
            del self.vad_residual[:frame_size]

            # # Begin at 90% of the minimum chunk size is filled
            # vad_start_threshold = int(self.min_samples * 0.90)
            # is_speech = self.vad(frame, return_seconds=False) if len(self.speech_buffer) >= vad_start_threshold else True

            # Start VAD slightly before reaching the minimum chunk size
            silence_length_samples = int(self.sample_rate * self.silence_length_ms / 1000)
            start_vad_after = self.min_samples - silence_length_samples
            is_speech = self.vad(frame, return_seconds=False) if len(self.speech_buffer) >= start_vad_after else True

            if is_speech:
                self.speech_buffer.extend(frame)
                self.silence_duration_tracker = 0
            else:
                self.silence_duration_tracker += frame_duration_ms
                self.speech_buffer.extend(frame)

            long_enough = len(self.speech_buffer) >= self.min_samples
            enough_silence = self.silence_duration_tracker >= self.silence_length_ms 


            if (enough_silence and long_enough) or len(self.speech_buffer) >= self.max_samples: # Enough silence or max chunk size reached
                keep_for_next = self.speech_buffer[-self.overlap:] if self.overlap > 0 else []
                chunk = np.array(self.speech_buffer)

                print(f"length: {len(chunk)/self.sample_rate:.2f}") 

                self.speech_buffer = list(keep_for_next)
                transcript_out = self.transcribe_buffer(chunk)
                break

        return transcript_out

    def transcribe_buffer(self,chunk):
        raise TypeError("Implement transcribe with language specific model")
    
    # Should be called at the end of transcription to flush any remaining audio
    def flush(self):
        if self.vad_residual:
            self.speech_buffer.extend(self.vad_residual)
            self.vad_residual.clear()

        if self.speech_buffer:
            chunk = np.array(self.speech_buffer)
            self.speech_buffer.clear()
            # print(f"Flushing remaining audio of length: {len(chunk)/self.sample_rate:.2f} seconds")
            return self.transcribe_buffer(chunk)

        return None

class WhisperTranscriber(LiveTranscriber):
    def __init__(self, model_type = "base.en", device = "cpu",
                 sample_rate = 16000, min_chunk_duration_ms = 4500,
                 vad_threshold = 0.65, silence_length_ms = 550, overlap_length = 0.0):

        super().__init__(sample_rate = sample_rate, min_chunk_duration_ms = min_chunk_duration_ms,
             vad_threshold = vad_threshold, silence_length_ms = silence_length_ms, overlap_length = overlap_length, device = device)

        # Loading STT model
        print(f"Loading Faster Whisper model: {model_type} …")
        self.model = WhisperModel(model_type, compute_type="auto", device=device)
        print("Whisper model loaded!")

        self.transcript_context = ""

    def transcribe_buffer(self,chunk):
        if len(chunk) == 0:
            return None
        segments, _ = self.model.transcribe(chunk, initial_prompt=self.transcript_context)
        text = " ".join(s.text.strip() for s in segments)
        if text:
            self.transcript_context = text
            print(f"Transcribed text: {text}")
        return text

class DanishTranscriber(LiveTranscriber):
    def __init__(self, device = "cpu",
                 sample_rate = 16000, min_chunk_duration_ms = 4500, 
                 vad_threshold = 0.65, silence_length_ms = 550, overlap_length = 0.5):

        super().__init__(sample_rate = sample_rate, min_chunk_duration_ms = min_chunk_duration_ms,
             vad_threshold = vad_threshold, silence_length_ms = silence_length_ms, overlap_length = overlap_length, device = device)

        # Loading STT model
        print(f"Loading Danish transcriber model…")
        model = "CoRal-project/roest-wav2vec2-315m-v2"
        self.model = pipeline("automatic-speech-recognition", model=model)
        print("Danish transcriber model loaded")
        self.prev_chunk = []
    
    def transcribe_buffer(self, chunk):
        if len(chunk) == 0:
            return None
        
        text_timestamp = self.model(chunk, return_timestamps="word")
        text_timestamp['text'] = text_timestamp['text'].split(' ') # Making text into list

        if text_timestamp['chunks'][0]['timestamp'][0] <= 0.3: # Checking if word close to start boundary
            text_timestamp['text'].pop(0)
            text_timestamp['chunks'].pop(0)
        if len(text_timestamp['chunks'])>0:
            if text_timestamp['chunks'][-1]['timestamp'][1] >= (len(chunk)/self.sample_rate)-0.3: # Checking if word close to end of boundary
                text_timestamp['text'].pop(-1)
                text_timestamp['chunks'].pop(-1)
        
        for i,word in enumerate(text_timestamp['text']):
            if len(word)==1 and word!='i' and word!='ø' and word!='ø':
                text_timestamp['text'].pop(i)

        for i in range(len(text_timestamp['text']),0,-1): # Going backward to find max amount of duplicated words if any exist
            if len(self.prev_chunk)>=i:
                if text_timestamp['text'][0:i] == self.prev_chunk[-i:]:
                    text_timestamp['text'] = text_timestamp['text'][i:]
                    break

        self.prev_chunk = text_timestamp['text']
        text = " ".join(text_timestamp['text'])
        print(text)
        return text

