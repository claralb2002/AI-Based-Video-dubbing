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
    def stt_worker(audio_queue, text_queue, wf_channels):
        import whisper # imports inside to avoid problems when using multi processing
    
        model = whisper.load_model("base", device="cpu") #loading Whisper model on CPU
        
        
        while True:
            audio_chunk = audio_queue.get() # get audio chunk from queue, and waits if it gets nothing
            if audio_chunk is None: # If None, it stops
                text_queue.put(None) # pass None to translator
                break
            
            audio_samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) #Convert audio bytes to a NumPy array of floats
            
            if audio_samples.ndim == 1 and wf_channels == 2: # check if its a 1D but actually from 2 channels
                audio_samples = audio_samples.reshape(-1, 2).mean(axis=1) # reshape and average across channels
            
            
            audio_samples = audio_samples / 32768.0 #normalize between -1 and 1. Divide by the max value for int16
            
            result = model.transcribe(audio_samples, language="en") # Do the actual transcription "en" = english
        
            transcription = result['text'].strip() #get text and clean up for extra spaces
            text_queue.put(transcription) # put transcribed text into the queue for translation



        
 
        
    
    
    # Worker process for Translation using MarianMT (Helsinki-NLP)
    def translation_worker(text_queue, output_queue=None):
        from transformers import pipeline # uses Hugging Face transformers for translation
        
        translator = pipeline("translation_en_to_da", model="Helsinki-NLP/opus-mt-en-da", device=-1) # device = -1 means using CPU. 
        
        while True:
            text = text_queue.get() # get text from the input queue. This will wait if queue is empty.
            if text is None: # If None, it stops
                break # exit loop
            
            
            translated_text = translator(text)[0]['translation_text'].strip() # get translated text and remove any leading whitespace
            
            print(f"Translated: {translated_text}") # printing to the console
            
            if output_queue: # if an output queue was provided
                output_queue.put(translated_text) # put translated text into that queue
    
    
    
    
    
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
    
