from transformers import MarianTokenizer, MarianMTModel

class ChunkTranslator:
    def __init__(self, model_name, context_length=2, num_beams=2):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.context_length = context_length
        self.delimeter = " <sep> "
        self.context_buffer = []
        self.num_beams = num_beams
    
    def translate_chunk(self, chunk):
        chunk = chunk.rstrip()
        self.context_buffer.append(chunk)

        if len(self.context_buffer) > self.context_length:
            self.context_buffer.pop(0)
        
        input_text = self.delimeter.join(self.context_buffer)
        input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=False, truncation=False)
        output_tokens = self.model.generate(**input_tokens, num_beams=self.num_beams, early_stopping=True)
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()
        output_text = output_text.replace("<sep >", "<sep>")


        if self.context_length > 1 and self.delimeter in output_text:
            parts = output_text.split(self.delimeter)
            translated_chunk = parts[-1].strip()
        else:
            translated_chunk = output_text
        return translated_chunk


    def reset_context(self):
        self.context_buffer = []


# if __name__ == "__main__":
#     translator = ChunkTranslator("Helsinki-NLP/opus-mt-en-da", context_length=2, num_beams=2)
#     text_chunks = [
#         "I met john yesterday",
#         "we went to the",
#         "park and had a great",
#         "time together. It was",
#         "a sunny day and we enjoyed",   
#         "the fresh air and the",
#     ]

#     for chunk in text_chunks:
#         translation = translator.translate_chunk(chunk)
#         print(f"Input: {chunk}")
#         print(f"Translated: {translation}\n")