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
        
global_cfg = read_config('global-config.yml')
file_extensions = ['.wav', '.mp3', '.m4a']

if __name__ == '__main__':
    input_dirs = glob(str(Path(global_cfg['input_dir'], '*')) + '/')
    dirs_to_process = [d for d in input_dirs if not os.path.isfile(Path(d, 'processed.txt'))]

    print(f'\nfound {len(dirs_to_process)} directories to process...\n')

    for directory in dirs_to_process:
        
        # identify relevant files
        files = [Path(directory, file) for file in os.listdir(directory)]
        files_to_process = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[-1] in file_extensions]
        
        # set configuration 
        cfg = global_cfg
        local_cfg_files = glob(str(Path(directory, 'config.yml')))
        if local_cfg_files:
            cfg.update(read_config(local_cfg_files[0]))
        
                
        # process files
        for file in files_to_process:
            
            # transcribe
            subprocess.run(
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
                    '--compute_type', 'int8'
                ]
                )
            
            
            # add to processed file
            with open(Path(directory, 'processed.txt'), 'a') as processed_file:
                processed_file.writelines([str(file)])
            
            # densify vtt output
            vtt_file = str(file)[:-4] + '.vtt'
            vtt_to_dense_vtt(vtt_file)
