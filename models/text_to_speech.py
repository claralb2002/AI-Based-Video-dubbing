import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pathlib import Path
import numpy as np
from utils.danish_replacement import replace_danish_letters
from datasets import load_dataset

"""
SpeechT5 from Microsoft

https://huggingface.co/microsoft/speecht5_tts

Licence under MIT
"""

class SpeechT5:
    def __init__(self, model_id="microsoft/speecht5_tts", device="cpu"):
        self.device = device
        print(f"Loading SpeechT5 model: {model_id}")
        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        self.sample_rate = self.vocoder.config.sampling_rate
        print("SpeechT5 model loaded!")

        # Load embedding
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    def speak(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Generate audio waveform from input text
        with torch.no_grad():
            waveform = self.model.generate(
                input_ids,
                speaker_embeddings=self.speaker_embedding,
                vocoder=self.vocoder
            )

        audio = waveform.cpu().numpy().squeeze()
        return audio

"""
Danish SpeechT5 TTS model
JackismyShephard/speecht5_tts-finetuned-nst-da

Fined-tuned model of Microsoft's SpeechT5 model for Danish text-to-speech synthesis.

https://huggingface.co/JackismyShephard/speecht5_tts-finetuned-nst-da
https://github.com/JackismyShephard/hugging-face-audio-course/blob/main/notebooks/inference/finetuned-nst-da-inference.ipynb

Licence under MIT
"""

class DanishSpeechT5:
    def __init__(self, model_id="JackismyShephard/speecht5_tts-finetuned-nst-da", embedding_path="../utils/male_51_vest_sydsjaelland.npy", device="cpu"):
        self.device = device
        print(f"Loading SpeechT5 Danish model: {model_id}")
        self.processor = SpeechT5Processor.from_pretrained(model_id)               
        self.model     = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(device)     
        self.vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        self.sample_rate = self.vocoder.config.sampling_rate                   
        print("Danish SpeechT5 model loaded!")

        # Load the Danish speaker embedding 
        embedding_np = np.load(Path(embedding_path))
        self.speaker_embedding = torch.tensor(embedding_np, dtype=torch.float).unsqueeze(0).to(device) 


    def speak(self, text):
        # Replace Danish letters with their English equivalents for the finetuned Danish model
        text = replace_danish_letters(text)

        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Generate audio waveform from input text
        with torch.no_grad():
            waveform = self.model.generate(input_ids,
                                           speaker_embeddings=self.speaker_embedding,
                                           vocoder=self.vocoder)
       
        audio = waveform.cpu().numpy().squeeze()
        return audio


   
