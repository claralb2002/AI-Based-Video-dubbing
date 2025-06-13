import sounddevice as sd
from pydub import AudioSegment
import time, sys
from pipeline import Pipeline
from threading import Thread, Event

FRAME_RATE   = 16000
CHANNELS     = 1
BLOCK_FRAMES = 512          # 32 ms at 16 kHz
DTYPE        = 'int16'       # 2-byte samples → AudioSegment sample_width = 2

stop_event = Event()         # lets us shut everything down cleanly

def print_outputs(output_queue):
    while not stop_event.is_set():
        result = output_queue.get()
        if result is None:           # sentinel value from Pipeline.stop()
            break
        translated, audio = result
        print(f"TRANSLATED: {translated}")
        sd.play(audio, FRAME_RATE)
        sd.wait()

def record_audio(audio_queue):
    """Continuously capture mic audio and push small AudioSegments to the queue."""

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        raw = indata.copy().tobytes()
        segment = AudioSegment(
            raw,
            frame_rate=FRAME_RATE,
            channels=CHANNELS,
            sample_width=2     
        )
        audio_queue.put(segment)

    with sd.InputStream(
            samplerate=FRAME_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_FRAMES,
            callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

if __name__ == "__main__":
    pipeline = Pipeline(input_language="en", output_language="da")
    pipeline.start()

    printer_thread  = Thread(target=print_outputs,  args=(pipeline.output_queue,))
    recorder_thread = Thread(target=record_audio,  args=(pipeline.audio_queue,))

    printer_thread.start()
    recorder_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Finished")
        stop_event.set()            # tell threads & stream loops to exit
        pipeline.stop()             # put sentinel into output_queue

        printer_thread.join()
        recorder_thread.join()
