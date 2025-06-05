from transformers import VitsModel, AutoTokenizer
import sounddevice as sd
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pathlib import Path
import numpy as np

"""
Meta's MMS (Multilingual Speech Synthesis)

https://huggingface.co/docs/transformers/en/model_doc/mms
"""

class MMS_speaker:
    def __init__(self, model_id="facebook/mms-tts-eng"):
        print(f"Loading MMS model: {model_id}")
        self.model = VitsModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.sample_rate = self.model.config.sampling_rate
        print("MMS speaker model loaded!")

    def speak(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            audio_waveform = self.model(**inputs).waveform

        audio = audio_waveform.cpu().numpy().squeeze()
        return audio

        #sd.play(audio, self.sample_rate)
        #sd.wait()

"""
Danish SpeechT5 TTS model
JackismyShephard/speecht5_tts-finetuned-nst-da

Fined tuned model of Microsofts SpeechT5 model for Danish text-to-speech synthesis.

https://huggingface.co/JackismyShephard/speecht5_tts-finetuned-nst-da
https://github.com/JackismyShephard/hugging-face-audio-course/blob/main/notebooks/inference/finetuned-nst-da-inference.ipynb
"""

class DanishSpeechT5:
    def __init__(self, model_id="JackismyShephard/speecht5_tts-finetuned-nst-da", embedding_path="../utils/male_51_vest_sydsjaelland.npy"):
        
        print(f"Loading SpeechT5 Danish model: {model_id}")
        self.processor = SpeechT5Processor.from_pretrained(model_id)               
        self.model     = SpeechT5ForTextToSpeech.from_pretrained(model_id)         
        self.vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.sample_rate = self.vocoder.config.sampling_rate                   
        print("Danish SpeechT5 model loaded!")

        # load the fixed speaker embedding (from: JackismyShephard/embeddings/nst-da-metricgan-plus/male_51_vest_sydsjaelland.npy Github)
        embedding_np = np.load(Path(embedding_path))
        self.speaker_embedding = torch.tensor(embedding_np, dtype=torch.float).unsqueeze(0)  


    def speak(self, text):
        inputs = self.processor(text=text, return_tensors="pt")

        with torch.no_grad():
            waveform = self.model.generate(inputs["input_ids"],
                                           speaker_embeddings=self.speaker_embedding,
                                           vocoder=self.vocoder)
       
        audio = waveform.cpu().numpy().squeeze()
        return audio

        #sd.play(audio, self.sample_rate)
        #sd.wait()

