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
def setup_latency_logger(log_dir = "logs", file_name = f"latency_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

    log_path = log_dir + "/" + file_name + ".csv"

    logger = logging.getLogger("latency_logger")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # write header if file doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("time,chunk_id,stt_latency_ms,tt_latency_ms,tts_latency_ms,total_latency_ms\n")

        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # handler for log files
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # handler for console output
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger