import re
from num2words import num2words

def expand_abbreviations(text, lang='en'):
    def abbr_replacer(match):
        # Split capital matches. 'PR' -> 'P R'
        token = match.group()
        if token.isalpha() and token.isupper():
            return ' '.join(char for char in token)

        # Split if containing numbers in abbrev. 'co2' -> 'C O two'
        expanded = []
        for char in token:
            if char.isdigit():
                expanded.append(num2words(int(char), lang=lang))
            else:
                expanded.append(char.upper()) 
        return ' '.join(expanded)

    return re.sub(r'\b(?:[A-Z]{2,}|[a-zA-Z]*\d+[a-zA-Z]*)\b', abbr_replacer, text)

def numbers_to_words(text, lang='en', split_abbreviations=True):
    def replacer(match):
        num_str = match.group()
        try:
            # Normalize spaces ("1910 erne" → "1910erne")
            num_str = num_str.replace(" ", "")

            # Percentages
            if '%' in num_str:
                number = int(num_str.rstrip('%'))
                percent_word = "procent" if lang == "da" else "percent"
                return num2words(number, lang=lang) + f" {percent_word}"
            
            # Danish decades 
            elif ((num_str.endswith('s') and num_str[:-1].isdigit()) or 
                  (num_str.endswith('erne') and num_str[:-4].isdigit())):
                decade = int(num_str[:-1]) if num_str.endswith('s') else int(num_str[:-4])
                if decade < 2000:
                    hundreds = decade // 100
                    tens = decade % 100
                    if lang == "da":
                        hundreds_word = num2words(hundreds, lang=lang)
                        tens_word = num2words(tens, lang=lang)
                        return f"{hundreds_word} {tens_word}" + 'erne'
                    
                    # English decades
                    else:
                        hundreds_word = num2words(hundreds, lang=lang)
                        tens_word = num2words(tens, lang=lang)
                        if tens_word.endswith('y'):
                            tens_word = tens_word[:-1] + 'ies'
                        else:
                            tens_word += 's'
                        return f"{hundreds_word} {tens_word}"
                    
                # 2000s 
                else:
                    return num2words(decade, lang=lang) + ("s" if num_str.endswith('s') else "erne")

            # Convert thousands to hundreds
            elif num_str.isdigit() and len(num_str) == 4:
                year = int(num_str)
                if year < 2000:
                    hundreds = year // 100
                    tens = year % 100
                    hundreds_word = num2words(hundreds, lang=lang)
                    tens_word = f" {num2words(tens, lang=lang)}" if tens else ""
                    connector = "hundrede" if lang == "da" else "hundred"
                    return f"{hundreds_word} {connector}{tens_word}"
                
            elif num_str == "1" and lang == "da":
                return "en"
            
            else:
                return num2words(float(num_str), lang=lang)
        except:
            return num_str  

    pattern = r'\b\d+\s?%|\d+s\b|\b\d+\s?erne\b|\b\d+(\.\d+)?\b'
    processed_text = re.sub(pattern, replacer, text)
    
    if split_abbreviations:
        processed_text = expand_abbreviations(processed_text, lang=lang)

    return processed_text

if __name__ == "__main__":   
    print(numbers_to_words('1910erne co 2', lang='da', split_abbreviations=True))  # "nitten hundrede tallet"

