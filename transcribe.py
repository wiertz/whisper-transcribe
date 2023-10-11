# Wrapper for whisperx that automatically transcribes audio files in "input" folder

import yaml
import whisperx
from whisperx import utils
from pathlib import Path
from glob import glob
import os
import gc
import torch
from vtt_to_dense_vtt import vtt_to_dense_vtt

def read_config(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)
        
global_cfg = read_config('global-config.yml')

asr_options = {
    "beam_size": 5,
    "patience": None,
    "length_penalty": 1.0,
    "temperatures": 0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "initial_prompt": None,
    "suppress_tokens": [int(x) for x in "-1".split(",")],
    "suppress_numerals": True
}

vad_options = {
    "vad_onset": 0.500,
    "vad_offset": 0.363
}

file_extensions = ['.wav', '.mp3', '.m4a']

if __name__ == '__main__':
    input_dirs = glob(str(Path(global_cfg['input_dir'], '*')))
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
        
        print(f'processing {len(files_to_process)} file in {directory}')
        
              
        
        
        # output writer
        write_result = utils.get_writer('vtt', directory)
        
        # process files
        for file in files_to_process:
            
            # load audio from file
            audio = whisperx.load_audio(file)
            
            # transcribe
            model = whisperx.load_model(
                cfg['whisper_model'], 
                cfg['torch_device'], 
                device_index = 0,
                compute_type=cfg['compute_type'], 
                language=cfg['language'], 
                asr_options=asr_options,
                vad_options=vad_options,
                task='transcribe',
                threads=4)
            
            result = model.transcribe(audio, batch_size=1)
            del(model)
            gc.collect()
            torch.cuda.empty_cache()
            
            # align
            model_a, metadata = whisperx.load_align_model(language_code=cfg['language'], device=cfg['torch_device'])
            result = whisperx.align(result['segments'], model_a, metadata, audio, cfg['torch_device'], return_char_alignments=False)
            del(model_a)
            gc.collect()
            torch.cuda.empty_cache()
            
            # diarize
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=cfg['huggingface_token'], device=cfg['torch_device'])
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            result['language'] = cfg['language']
            del(diarize_model)
            gc.collect()
            torch.cuda.empty_cache()
            
            
            # write result
            write_result(result, file, {'highlight_words': False, "max_line_count": None, "max_line_width": None})
            
            with open(Path(directory, 'processed.txt'), 'w') as processed_file:
                processed_file.writelines([str(file)])
            
            # densify vtt output
            vtt_file = file + '.vtt'
            vtt_to_dense_vtt(vtt_file)
