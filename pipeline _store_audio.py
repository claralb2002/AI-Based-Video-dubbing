from multiprocessing import Process, Queue
from utils.audio_preprocessing import preprocess_audio
from utils.audio_streaming import stream_audio
import sounddevice as sd
    
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

            result = model.add_audio_chunk(chunk)
            if result:
                transcription_queue.put(result)

        final_result = model.flush()
        if final_result:
            transcription_queue.put(final_result)

        transcription_queue.put(None)
        
    
    @staticmethod
    def translation_worker(transcription_queue, translation_queue, input_language):
        from models.translator import ChunkTranslator
        
        if input_language == "en":
            model = ChunkTranslator(model_name="Helsinki-NLP/opus-mt-en-da")
        if input_language == "da":
            model = ChunkTranslator(model_name="Helsinki-NLP/opus-mt-da-en")


        while True:
            transcription = transcription_queue.get()
            if transcription is None:
                break
            translated = model.translate_chunk(transcription)
            translation_queue.put(translated)
    
    @staticmethod
    def tts_worker(translation_queue, output_queue, output_language, device="cpu"):
        from models.text_to_speech import DanishSpeechT5, SpeechT5
        
        if output_language == "da":
            model = DanishSpeechT5(embedding_path="utils/male_51_vest_sydsjaelland.npy", device=device)
        if output_language == "en":
            model = SpeechT5(device=device)    
        
        while True:
            translation = translation_queue.get()
            if translation is None:
                break
            audio = model.speak(translation)
            output_queue.put((translation, audio))


    def start(self):
        self.stt_proc = Process(target=self.stt_worker, args=(self.audio_queue, self.transcription_queue, self.input_language))
        self.trans_proc = Process(target=self.translation_worker, args=(self.transcription_queue, self.translation_queue, self.input_language))
        self.tts_proc = Process(target=self.tts_worker, args=(self.translation_queue, self.output_queue, self.output_language, self.device))
        self.stt_proc.start()
        self.trans_proc.start()
        self.tts_proc.start()

    def stop(self):
        self.audio_queue.put(None)
        self.stt_proc.join()
        self.trans_proc.join()
        self.tts_proc.join()

# utils/audio_preprocessing.py
import numpy as np

from moviepy import VideoFileClip

def audio_from_video(video_path: str,
                     target_sr: int = 16_000) -> np.ndarray:
    """
    Extract and resample mono audio from a video file.
    Returns float32 PCM in [-1, 1] at `target_sr`.
    """
    clip = VideoFileClip(video_path)
    audio = clip.audio.to_soundarray(fps=target_sr)

    # stereo ➜ mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # ensure MoviePy closes the file handles promptly
    clip.close()

    return audio.astype("float32")



if __name__ == "__main__":
    from threading import Thread
    from utils.audio_preprocessing import preprocess_audio

    # ---------- choose the media file ----------
    video_path = "IMG_3250.MOV"          # your iPhone clip

    use_video = True                         # flip this

    input_language  = "en"
    output_language = "da"

    # ---------- load and prepare ----------
    if use_video:
        audio = audio_from_video(video_path, target_sr=16_000)

    # ---------- start the pipeline ----------
    pipeline = Pipeline(input_language=input_language,
                        output_language=output_language,
                        device="cpu")
    pipeline.start()