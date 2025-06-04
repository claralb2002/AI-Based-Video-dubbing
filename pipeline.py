from models.whisper_transcriber import WhisperTranscriber
from models.translator import ChunkTranslator

import time
import wave
import numpy as np
from multiprocessing import Process, Queue



class Pipeline:
    def __init__(self, wav_file, chunk_duration=5.0):
        self.wav_file = wav_file
        self.chunk_duration = chunk_duration
        self.audio_queue = Queue()
        self.text_queue = Queue()
        # self.tts_queue = Queue() 

    # Worker process for Speech-to-Text using OpenAI Whisper
    def stt_worker(audio_queue):
        import whisper # imports inside to avoid problems when using multi processing
    
        model = whisper.load_model("base", device="cpu") #loading Whisper model on CPU
        
        
 
        
    
    
    # Worker process for Translation using MarianMT (Helsinki-NLP)
    def translation_worker(text_queue, output_queue=None):
        pass
    
    
    
    
    
    # Audio streaming simulation and pipeline orchestration
    def run(self):
        with wave.open(self.wav_file, 'rb') as wf: # 'rb' for read binary
            wf_channels = wf.getnchannels() # number of audio channels
            frame_rate = wf.getframerate() # sample rate
            chunk_frames = int(frame_rate * self.chunk_duration) # Calculate chunk size in frames based on wished duration. This is how many frames we read at once, makes sense

            # Using models
            stt_process = Process(target=self.stt_worker, args=(self.audio_queue, self.text_queue, wf_channels)) # Passing queues and channel info
            trans_process = Process(target=self.translation_worker, args=(self.text_queue, None)) # need text queue for translation
            stt_process.start() # start processes
            trans_process.start() # start processes


            while True: # loop until we hit the end of the file
                data = wf.readframes(chunk_frames) # read audio data in chunks
                if len(data) == 0: # break when we reach the end
                    break
                self.audio_queue.put(data) # put audio data into the queue for STT
                time.sleep(chunk_frames / float(frame_rate)) #make it flow realtistisc
            self.audio_queue.put(None) # Signal end of STT worker by putting None

            stt_process.join() # wait for STT to complete
            trans_process.join() # wait for translation to complete




# --- Run pipeline ---
if __name__ == "__main__":
    wav_path = "/data/brownie_beaver_4_min.wav"
    pipeline = Pipeline(wav_file=wav_path)
    pipeline.run()
    
