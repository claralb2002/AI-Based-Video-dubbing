<table style="width:100%; border-collapse:collapse; table-layout:fixed;">
  <tr>
    <td style="width:80%; text-align:left; vertical-align:top; padding:0;">
      <h1 style="margin:0;">Live AI Translation</h1>
      <h2 style="margin:0; font-weight:normal;">Real-Time Multilingual Communication</h2>
    </td>
    <td style="width:20%; text-align:right; vertical-align:top; padding:0;">
      <img src="DTULogo.png" alt="DTU Logo" style="height:80px; display:block; margin-left:auto;" />
    </td>
  </tr>
</table>


![App Screenshot](https://cdn.prod.website-files.com/61707b4f874fa22b7482b07e/647e84a46f030f90ae924352_In-person%20Virtual%20Hello-p-1080.png)
[link](https://cdn.prod.website-files.com/61707b4f874fa22b7482b07e/647e84a46f030f90ae924352_In-person%20Virtual%20Hello-p-1080.png)


This project explores the development of a real-time audio translation system by integrating speech-to-text, text-translation, and text-to-speech components into a low-latency, high-accuracy processing pipeline.


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



## Data structure

`data/`: Contains all sound files and their corresponding transcriptions in  both English and Danish.

`logs/`: Stores log files generated during pipeline execution, including performance metrics and processing details







## License

[MIT](https://choosealicense.com/licenses/mit/)


## Authors

- Clara
- Mads
- Julius
- Joseph

