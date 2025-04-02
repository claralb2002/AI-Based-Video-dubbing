from pydub import AudioSegment

def preprocess_audio(audio_file, target_rate=16000):
    """
    Loads an audio file and preprocesses it to meet Whisper's requirements.

    Parameters:
        audio_file (str): Path to the audio file.
        target_rate (int): Target sample rate (default: 16000 Hz).

    Returns:
        AudioSegment: Processed audio.
    """
    
    audio = AudioSegment.from_file(audio_file)
    return audio.set_frame_rate(target_rate).set_channels(1).set_sample_width(2)