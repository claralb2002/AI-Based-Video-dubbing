<div style="display: flex; align-items: start;  width: 100%;">
  <img src="readme_images/DTULogo.png" alt="DTU Logo" style="height: 80px;" />
  <div>
    <h1 style="margin: 0;">Live AI Translation</h1>
    <h2 style="margin: 0; font-weight: normal;">Real-Time Multilingual Communication</h2>
  </div>
</div>



![App Screenshot](https://cdn.prod.website-files.com/61707b4f874fa22b7482b07e/647e84a46f030f90ae924352_In-person%20Virtual%20Hello-p-1080.png)
[link to image](https://cdn.prod.website-files.com/61707b4f874fa22b7482b07e/647e84a46f030f90ae924352_In-person%20Virtual%20Hello-p-1080.png)

## Introduction

This project presents a real-time, multilingual audio translation system that enables seamless speech-to-speech translation. It is especially designed for live use cases such as academic lectures, international meetings, or global events. The system converts live audio input into translated speech using a pipeline that integrates:

<ul>
  <li>Speech-to-Text (STT)</li>
  <li>Text Translation (TT)</li>
  <li>Text-to-Speech (TTS)</li>
</ul>

The result is a live dubbing tool capable of bridging language barriers in real time.

<img src="readme_images/simple_pipeline.png" alt="Simple Pipeline" style="width: 100%;" />



## Features
- Live audio translation via microphone
- Support for both English -> Danish and Danish -> English
- Detailed logging and performance metrics
- Streaming architecture with multiprocessing queues
- Real-time latency tracking
- Graph and CSV output generation



## Requirements
- Python 3.10+ 
- Conda or Mamba
- Microphone (for live input)


## File Descriptions
 <ul>
    <li><code>pipeline.py</code> – Processes audio files</li>
    <li><code>pipeline_demo.py</code> – Enables live microphone input</li>
    <li><code>pipeline_wlog.py</code> – Same as <code>pipeline.py</code> but with detailed logs</li>
    <li><code>processing_results.ipynb</code> – Generates evaluation plots, tables, and CSVs</li>
    <li><code>environment.yml</code> – Conda environment with dependencies</li>
    <li><code>data/</code> – Audio samples + transcriptions</li>
    <li><code>logs/</code> – Runtime logs with latency and transcription data</li>
    <li><code>results/</code> – Evaluation outputs: WER, CER, COMET, latency</li>
    <li><code>images/</code> – Saved plots and diagrams</li>
  </ul>






## Installation of environemnt

For setup run
```bash
  conda env create -f environment.yml # or mamba
  conda activate '02466_AI_dubbing'
  python -m ipykernel install --user --name='02466_AI_dubbing' --display-name "Python ('02466_AI_dubbing')"
```

To update env, run the following
```bash
    conda env update -f environment.yml --prune

```



## How to use

<b>Run the pipeline on audio files:</b> Execute `pipeline.py`to process pre-recorded sound files through the full dubbing pipeline.

<b> Use your microphone for live input:</b> Execute `pipeline_demo.py`to capture audio from your microphone and process it in real-time.

<b> Generate detailed logs:</b> Execute `pipeline_wlog.py`to process audio files and generate a CSV log for each chunk.

<b> Generate results: </b> Execute `processing_results.ipynb` to generate results CSV, plots and table.

## Technologies Used

  <ul>
    <li><a href="https://github.com/openai/whisper">Whisper</a> (English STT)</li>
    <li><a href="https://huggingface.co/CoRal-project/roest-wav2vec2-315m-v2">RØST wav2vec2</a> (Danish STT)</li>
    <li><a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-da">MarianMT</a> (Text Translation)</li>
    <li><a href="https://huggingface.co/microsoft/speecht5_tts">SpeechT5</a> (Text-to-Speech)</li>
    <li><a href="https://github.com/snakers4/silero-vad">Silero VAD</a> (Voice Activity Detection)</li>
    <li><a href="https://github.com/huggingface/transformers">HuggingFace Transformers</a></li>
    <li><a href="https://github.com/openai/whisper">Faster-Whisper</a> via CTranslate2</li>
  </ul>


## Future Work


 <ul>
    <li>Support for additional languages (e.g., Spanish, German, French)</li>
    <li>Global loading </li>
    <li>Local cashing </li>
    <li>More robustness (dynamic silence length and VAD threshold) </li>
    <li>Handle larger pauses </li>
  </ul>


## Authors

- Clara Louise Brodt
- Joseph An Nguyen
- Julius Winkel
- Mads Helle Højgaard


## License

[MIT](https://choosealicense.com/licenses/mit/)

