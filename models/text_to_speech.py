from transformers import VitsModel, AutoTokenizer
import sounddevice as sd
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pathlib import Path
import numpy as np
from utils.danish_replacement import replace_danish_letters
from datasets import load_dataset


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

    def speak(self, text, speed=1.0):
        inputs = self.tokenizer(text, return_tensors="pt")
        self.model.speaking_rate = speed
        with torch.no_grad():
            audio_waveform = self.model(**inputs).waveform

        audio = audio_waveform.cpu().numpy().squeeze()
        return audio

        # sd.play(audio, self.sample_rate)
        # sd.wait()

"""
SpeechT5 from Microsoft
https://huggingface.co/microsoft/speecht5_tts
"""

class SpeechT5:
    def __init__(self, model_id="microsoft/speecht5_tts"):
        print(f"Loading SpeechT5 model: {model_id}")
        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.sample_rate = self.vocoder.config.sampling_rate
        print("SpeechT5 model loaded!")

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def speak(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            waveform = self.model.generate(
                inputs["input_ids"],
                speaker_embeddings=self.speaker_embedding,
                vocoder=self.vocoder
            )

        audio = waveform.cpu().numpy().squeeze()
        # sd.play(audio, self.sample_rate)
        # sd.wait()
        return audio

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


    def speak(self, text, speed=1.0):
        # Replace Danish letters with their English equivalents for the finetuned danish model
        text = replace_danish_letters(text)

        inputs = self.processor(text=text, return_tensors="pt")
    
        with torch.no_grad():
            waveform = self.model.generate(inputs["input_ids"],
                                           speaker_embeddings=self.speaker_embedding,
                                           vocoder=self.vocoder)
       
        if speed != 1.0:
            waveform = audio_speed_control(waveform, slowdown_factor=speed)

        audio = waveform.cpu().numpy().squeeze()
        # return audio

        sd.play(audio, self.sample_rate)
        sd.wait()
   

import torch.nn.functional as F
def audio_speed_control(audio_tensor, slowdown_factor=1.3):
    # length of the audio tensor
    length = audio_tensor.shape[-1]

    # calculate the new length based on the slowdown factor
    new_length = int(length * slowdown_factor)

    # interpoler for at strække waveformen
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    slowed = F.interpolate(audio_tensor, size=new_length, mode="linear", align_corners=True)
    return slowed.squeeze()

