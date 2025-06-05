from multiprocessing import Process, Queue
from utils.audio_preprocessing import preprocess_audio
from utils.audio_streaming import stream_audio
import sounddevice as sd
    
class Pipeline:
    def __init__(self):
        self.audio_queue = Queue()
        self.transcription_queue = Queue()
        self.translation_queue = Queue()
        self.output_queue = Queue()

    @staticmethod
    def stt_worker(audio_queue, transcription_queue):
        from models.whisper_transcriber import WhisperTranscriber

        model = WhisperTranscriber()

        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break

            result = model.add_audio_chunk(chunk)
            if result:
                transcription_queue.put(result)

        final_result = model.flush()
        if final_result:
            transcription_queue.put(final_result)

        transcription_queue.put(None)
        
    
    @staticmethod
    def translation_worker(transcription_queue, translation_queue):
        from models.translator import ChunkTranslator
        model = ChunkTranslator(model_name="Helsinki-NLP/opus-mt-en-da")

        while True:
            transcription = transcription_queue.get()
            if transcription is None:
                break
            translated = model.translate_chunk(transcription)
            translation_queue.put(translated)
    
    @staticmethod
    def tts_worker(translation_queue,output_queue):
        from models.text_to_speech import DanishSpeechT5
        model = DanishSpeechT5()
        while True:
            translation = translation_queue.get()
            if translation is None:
                break
            audio = model.speak(translation)
            output_queue.put((translation, audio))


    def start(self):
        self.stt_proc = Process(target=self.stt_worker, args=(self.audio_queue, self.transcription_queue))
        self.trans_proc = Process(target=self.translation_worker, args=(self.transcription_queue, self.translation_queue))
        self.tts_proc = Process(target=self.tts_worker, args=(self.translation_queue, self.output_queue))
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

    wav_path = "data/speaker_3.wav"
    audio = preprocess_audio(wav_path)

    pipeline = Pipeline()
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

    for chunk in stream_audio(audio, frame_ms=200):
        pipeline.audio_queue.put(chunk)

    pipeline.stop()
    printer_thread.join()




