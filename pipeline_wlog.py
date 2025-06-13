from multiprocessing import Process, Queue
from utils.audio_preprocessing import preprocess_audio
from utils.audio_streaming import stream_audio
import sounddevice as sd
import logging
import time
from datetime import datetime

""" 
Time latency logger for the pipeline:

This module logs the time taken for each stage of the pipeline: STT, translation, and TTS. As well as total time for a chunk, from input to output.

For each procces in the pipeline the input and output times of a chunk are logged, 
and the latencies are calculated for each chunk when they are fully processed througout the pipeline.
"""
# configure logging
logger = logging.getLogger("latency_logger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

# handler for log files
fh = logging.FileHandler(f"logs/latency_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
fh.setFormatter(formatter)
logger.addHandler(fh)

# handler for console output
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
    


"""
Pipeline class that orchestrates the STT, translation, and TTS processes:

The models (STT, TT, TTS) are implemented as classes, and imported from the models directory.
It uses multiprocessing to handle audio processing in parallel, allowing for real-time transcription, translation, and speech synthesis.

"""
class Pipeline:
    def __init__(self, input_language, output_language, device="cpu"):
        self.input_language = input_language
        self.output_language = output_language
        self.audio_queue = Queue()
        self.transcription_queue = Queue()
        self.translation_queue = Queue()
        self.output_queue = Queue()
        self.device = device

    @staticmethod
    def stt_worker(audio_queue, transcription_queue, input_language):
        from models.whisper_transcriber import WhisperTranscriber, DanishTranscriber
        if input_language == "en":
            model = WhisperTranscriber(model_type="base.en", device="cpu")
        if input_language == "da":
            model = DanishTranscriber()

        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break

            # Log the time when the audio chunk is received
            chunk_id, start_time, audio = chunk
            #logger.debug(f"Received chunk {chunk_id} at {start_time:.4f} seconds")  # ! this logs all samples as well (if commented out it will only log the final result)
            stt_start = time.perf_counter()
            result = model.add_audio_chunk(audio)
            stt_end = time.perf_counter()

            if result:
                transcription_queue.put((chunk_id, start_time, stt_start, stt_end, result))

        final_result = model.flush()
        if final_result:
            transcription_queue.put(final_result)

        transcription_queue.put(None)
        
    
    @staticmethod
    def translation_worker(transcription_queue, translation_queue, input_language, output_language):
        from models.translator import ChunkTranslator
        from utils.num_to_words import numbers_to_words
        if input_language == "en":
            model = ChunkTranslator(model_name="Helsinki-NLP/opus-mt-en-da")
        if input_language == "da":
            model = ChunkTranslator(model_name="Helsinki-NLP/opus-mt-da-en")


        while True:
            chunk = transcription_queue.get()
            if chunk is None:
                break

            chunk_id, start_time, stt_start, stt_end, transcription = chunk

            tt_start = time.perf_counter()
            transcription = numbers_to_words(transcription, lang=input_language)
            translated = model.translate_chunk(transcription)
            translated = numbers_to_words(translated, lang=output_language)
            tt_end = time.perf_counter()
            translation_queue.put((chunk_id, start_time, stt_start, stt_end, tt_start, tt_end, translated))
    
    @staticmethod
    def tts_worker(translation_queue, output_queue, output_language, device="cpu"):
        from models.text_to_speech import DanishSpeechT5, MMS_speaker, SpeechT5
        
        if output_language == "da":
            model = DanishSpeechT5(embedding_path="utils/male_51_vest_sydsjaelland.npy", device=device)
        if output_language == "en":
            model = SpeechT5(device=device)    
        
        while True:
            chunk = translation_queue.get()
            if chunk is None:
                break

            chunk_id, start_time, stt_start, stt_end, tt_start, tt_end, translation = chunk

            tts_start = time.perf_counter()
            audio = model.speak(translation)
            tts_end = time.perf_counter()

            # compute latencies (end times - start times)
            stt_latency = (stt_end - stt_start) * 1000
            trans_latency = (tt_end - tt_start) * 1000
            tts_latency = (tts_end - tts_start) * 1000
            total_latency = (tts_end - start_time) * 1000

            # Log the latencies for the chunk
            logger.info(
                f"Chunk {chunk_id}: STT={stt_latency:.2f}ms, TT={trans_latency:.2f}ms, "
                f"TTS={tts_latency:.2f}ms, TOTAL={total_latency:.2f}ms"
            )

            output_queue.put((translation, audio))


    def start(self):
        self.stt_proc = Process(target=self.stt_worker, args=(self.audio_queue, self.transcription_queue, self.input_language))
        self.trans_proc = Process(target=self.translation_worker, args=(self.transcription_queue, self.translation_queue, self.input_language, self.output_language))
        self.tts_proc = Process(target=self.tts_worker, args=(self.translation_queue, self.output_queue, self.output_language, self.device))
        self.stt_proc.start()
        self.trans_proc.start()
        self.tts_proc.start()

    def stop(self):
        self.audio_queue.put(None)
        self.stt_proc.join()
        self.trans_proc.join()
        self.tts_proc.join()

 
if __name__ == "__main__":
    from threading import Thread

    input_language = "en"
    output_language = "da"


    if input_language == "da":
        wav_path = "data/danish/dk_speaker_3.wav"
    if input_language == "en":
        wav_path = "data/english/speaker_3_final.wav"


    audio = preprocess_audio(wav_path)

    pipeline = Pipeline(input_language=input_language, output_language=output_language, device="mps")
    pipeline.start()

    def print_outputs(queue):
        while True:
            result = queue.get()
            if result is None:
                break
            translated, audio = result
            # print(f"TRANSCRIPT: {transcript}")
            print(f"TRANSLATED: {translated}")
            sd.play(audio, 16000)
            sd.wait()

    printer_thread = Thread(target=print_outputs, args=(pipeline.output_queue,))
    printer_thread.start()

    for i, chunk in enumerate(stream_audio(audio, frame_ms=200)):
        start_time = time.perf_counter()

        # Log the time when the audio chunk is sent to the pipeline (each sample is enumerated and time is logged) (!OBS: this is samples, not chunks)
        # But the time is only logged for the last sample of each chunk, so it will not log all samples.
        pipeline.audio_queue.put((i, start_time, chunk))

        
    pipeline.stop()
    printer_thread.join()




