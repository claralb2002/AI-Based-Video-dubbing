from transformers import MarianTokenizer, MarianMTModel
import re

class ChunkTranslator:
    def __init__(self, model_name, context_length=2, num_beams=2):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.context_length = context_length
        self.delimeter = " <_extra_id_0_> "
        self.context_buffer = []
        self.num_beams = num_beams
    
    def translate_chunk(self, chunk):
        chunk = chunk.rstrip().lower()
        self.context_buffer.append(chunk)

        if len(self.context_buffer) > self.context_length:
            self.context_buffer.pop(0)
        
        cleaned_buffer = [chunk.strip(" -") for chunk in self.context_buffer]
        input_text = self.delimeter.join(cleaned_buffer)

        input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=False, truncation=False)
        max_length = input_tokens['input_ids'].shape[1]*2
        output_tokens = self.model.generate(**input_tokens, num_beams=self.num_beams, early_stopping=True, repetition_penalty=1.5, max_length = max_length)
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()
        output_text = re.sub(r"<\s*_?\s*extra\s*_?\s*id\s*_?\s*0\s*_?\s*>", self.delimeter, output_text)

        if self.context_length > 1 and self.delimeter in output_text:
            parts = output_text.split(self.delimeter)
            translated_chunk = parts[-1].strip()
        else:
            input_tokens = self.tokenizer(chunk, return_tensors="pt", padding=False, truncation=False)
            output_tokens = self.model.generate(**input_tokens, num_beams=self.num_beams, early_stopping=True, repetition_penalty=1.5, max_length = max_length)
            output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()
            translated_chunk = output_text
            if self.delimeter in input_text:
                print(f"Warning: No delimiter found in OUTPUT: {output_text} INPUT: {input_text}")
                print(f'translated_chunk: {translated_chunk}')
        return translated_chunk

    def reset_context(self):
        self.context_buffer = []



if __name__ == "__main__":
    translator = ChunkTranslator("Helsinki-NLP/opus-mt-da-en", context_length=2, num_beams=2)
    text_chunks = [
        "er den største årsag som folk skal bekymre sig om og hvis man går",
        "i dybten med det er det især tilfældet for mænd som man",
    ]

    for chunk in text_chunks:
        translation = translator.translate_chunk(chunk)
        print(f"Input: {chunk}")
        print(f"Translated: {translation}\n")