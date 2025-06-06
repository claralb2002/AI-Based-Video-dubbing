"""
This module provides a function to replace Danish letters and special characters, made by JackismyShephard.

https://github.com/JackismyShephard/hugging-face-audio-course/blob/main/notebooks/demos/finetuned-nst-da-demo.ipynb
"""

def replace_danish_letters(text):
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


replacements = [
    ("&", "og"),
    ("\r", " "),
    ("'", ""),
    ("\\", ""),
    ("¨", " "),
    ("Å", "AA"),
    ("Æ", "AE"),
    ("É", "E"),
    ("Ö", "OE"),
    ("Ø", "OE"),
    ("á", "a"),
    ("ä", "ae"),
    ("å", "aa"),
    ("è", "e"),
    ("î", "i"),
    ("ô", "oe"),
    ("ö", "oe"),
    ("ø", "oe"),
    ("ü", "y"),
]