from transformers import MarianTokenizer, MarianMTModel
class ChunkTranslator:
    def __init__(self, model_name, context_length=1, num_beams=2):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.context_length = context_length
        self.delimeter = "... "
        self.context_buffer = []
        self.num_beams = num_beams
    
    def translate_chunk(self, chunk):
        chunk = chunk.rstrip()
        self.context_buffer.append(chunk)

        if len(self.context_buffer) > self.context_length:
            self.context_buffer.pop(0)
        
        input_text = self.delimeter.join(self.past_context)
        input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=False, truncation=False)

        ouput_tokens = self.model.generate(**input_tokens, num_beams=self.num_beams, early_stopping=True)
        output_text = self.tokenizer.batch_decode(ouput_tokens, skip_special_tokens=True)[0]

        if self.context_length > 0 and len(self.context_buffer) > 1:
            output_text = output_text.split(self.delimeter)
            translated_chunk = output_text[-1].strip()
        else:
            translated_chunk = output_text.strip()
        
        return translated_chunk
    
    def reset_context(self):
        self.context_buffer = []
