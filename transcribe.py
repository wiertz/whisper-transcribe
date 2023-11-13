# Wrapper for whisperx that automatically transcribes audio files in "input" folder

import yaml
from pathlib import Path
from glob import glob
import os
import subprocess
from vtt_to_dense_vtt import vtt_to_dense_vtt

def read_config(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)
    
cwd = os.path.abspath(os.path.dirname(__file__))
        
global_cfg = read_config(Path(cwd, 'global-config.yml'))
ignore_extensions = ['.txt', '.yml', '.vtt']

if __name__ == '__main__':
    input_dirs = glob(str(Path(global_cfg['input_dir'], '*')) + '/')
    dirs_to_process = [d for d in input_dirs if os.path.isfile(Path(d, 'config.yml'))]

    print(f'\nfound {len(dirs_to_process)} directories to process...\n')

    for directory in dirs_to_process:
        
        # identify relevant files
        files = [Path(directory, file) for file in os.listdir(directory)]
        valid_files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[-1].lower() not in ignore_extensions]
        unprocessed_files = [f for f in files if not os.path.isfile(os.path.splitext(f)[0] + '.vtt')]
        
        # set configuration 
        cfg = global_cfg
        local_cfg_files = glob(str(Path(directory, 'config.yml')))
        if local_cfg_files:
            cfg.update(read_config(local_cfg_files[0]))
        
                
        # process files
        for file in unprocessed_files:
            
            # transcribe
            process = subprocess.run(
                [
                    cfg['whisperx_cmd'], 
                    f"{file}",
                    '--diarize',
                    '--model', cfg["whisper_model"],
                    '--language', cfg["language"],
                    '--hf_token', cfg["huggingface_token"],
                    '--output_dir', f"{directory}",
                    '--device', cfg["device"],
                    '--output_format', 'vtt',
                    '--compute_type', 'int8',
                    '--compression_ratio_threshold', '2',
                    '--no_speech_threshold', '0.5'
                ], check=False
                )
            
            
            # add to processed file
            with open(Path(directory, 'processed.txt'), 'a') as processed_file:
                if process.returncode != 0:
                    processed_file.write('ERROR ')
                
                processed_file.write(str(file))
                processed_file.write('\n')
            
            # densify vtt output
            vtt_file = str(file)[:-4] + '.vtt'
            vtt_to_dense_vtt(vtt_file)
