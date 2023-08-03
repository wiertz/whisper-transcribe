# whisper-transcribe
whisper transcription with speaker diarization

# Installation
Tested with Python 3.10.12. It is advised to run this in a new virtual environment to avoid dependency version conflicts. After activating the environment, run `pip install -r requirements.txt` to install all dependencies. (Installation of dependencies requires build tools to be available. On Linux/Ubuntu make sure to install `python3.10-dev` and `build-essential`).

# Usage
Change parameters at the top of transcribe.py (particularly the path to ffmpeg). Place audio files in subdirectory `input/`. Running `python transcribe.py` will create an output directory with transcripts. Processed files will be moved to `processed` (regardless of success or error).
