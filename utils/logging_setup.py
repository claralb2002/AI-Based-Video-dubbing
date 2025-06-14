import logging
import os
from datetime import datetime
"""

    Sets up a logger for latency logging.
    This logger will log the time, chunk ID, and latencies for STT, TT, TTS, and total latency in a CSV file.

    For each procces in the pipeline the input and output times of a chunk are logged, 
    and the latencies are calculated for each chunk when they are fully processed througout the pipeline.
    
"""

# for pipeline evaluation set file_name = input file + buffer size
def setup_latency_logger(file_name):

    log_path = "logs" + "/" + file_name + ".csv"

    logger = logging.getLogger("latency_logger")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # write header if file doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("time,chunk_id,stt_latency_ms,tt_latency_ms,tts_latency_ms,total_latency_ms,transcription,translation\n")

        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # handler for log files
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # handler for console output
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger