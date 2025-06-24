# 2. convert_mov_to_wav.py
import ffmpeg
from pathlib import Path

def mov_to_wav(input_path: str | Path, out_dir: str | Path | None = None):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = Path(out_dir) if out_dir else input_path.parent
    out_path = out_dir / (input_path.stem + ".wav")

    (
        ffmpeg
        .input(str(input_path))
        .output(str(out_path), acodec='pcm_s16le', ac=2, ar='48k')  # 16-bit PCM, stereo, 48 kHz
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path

if __name__ == "__main__":
    print("WAV written to:", mov_to_wav("IMG_3250.MOV"))