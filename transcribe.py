# Wrapper for whisperx that automatically transcribes audio files in "input" folder

import yaml
from pathlib import Path
from glob import glob
import os
import subprocess
import time
from vtt_to_dense_vtt import vtt_to_dense_vtt
from datetime import datetime


def read_config(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def transcribe_file(new_file):
        
    # set configuration 
    cfg = global_cfg
    parent_dir = os.path.dirname(new_file)
    local_cfg_files = glob(str(Path(parent_dir, 'config.yml')))
    if local_cfg_files:
        cfg.update(read_config(local_cfg_files[0]))

    # transcribe
    process = subprocess.run(
        [
            cfg['whisperx_cmd'], 
            f"{new_file}",
            '--diarize',
            '--model', cfg["whisper_model"],
            '--language', cfg["language"],
            '--hf_token', cfg["huggingface_token"],
            '--output_dir', f"{parent_dir}",
            '--device', cfg["device"],
            '--output_format', 'vtt',
            '--compute_type', 'int8',
            '--compression_ratio_threshold', '2',
            '--no_speech_threshold', '0.5'
        ], 
        check=False,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL
        )
    
    if process.returncode == 0:
        vtt_file = str(new_file)[:-4] + '.vtt'
        vtt_to_dense_vtt(vtt_file)

    else:
        with open(str(new_file)[:-4] + '.err', mode='w') as error_log:    
            error_log.write('Error: could not process audio file. Maybe audio file is corrupted?')

    return process.returncode
        
        
def is_new_audio_file(file, audio_extensions):
    if not os.path.splitext(file)[1].lower() in audio_extensions:
        return False
    
    if os.path.isfile(os.path.splitext(file)[0] + '.vtt'):
        return False
    
    if os.path.isfile(os.path.splitext(file)[0] + '.err'):
        return False
    
    return True


def find_unprocessed_files(dir, extensions):
    files_in_dir = glob(os.path.join(dir, '**'), recursive=True)
    new_audio_files = [f for f in files_in_dir if is_new_audio_file(f, extensions)]
    return new_audio_files
    

if __name__ == '__main__':     
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%m:%S")} Transcription process launched')
    cwd = os.path.abspath(os.path.dirname(__file__))
    global_cfg = read_config(Path(cwd, 'global-config.yml'))
    audio_extensions = ['.mp3', '.m4a', '.flac', '.mp4', '.wav', '.wma', '.aac', '.aiff', '.pcm', '.ogg', '.vobis']

    # process new files
    while True:
        from datetime import datetime
        unprocessed_files = find_unprocessed_files(global_cfg['input_dir'], audio_extensions)
        for f in unprocessed_files:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%m:%S")} Transcribing {f}')
            return_code = transcribe_file(f)
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%m:%S")}', end='')
            print(f'    ...OK') if return_code == 0 else print(print('    ...FAILED'))
        if not unprocessed_files:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%m:%S")} Waiting for new files', end='\r')
            time.sleep(60)
        
        
    
    