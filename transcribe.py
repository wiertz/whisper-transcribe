# Wrapper for whisperx that automatically transcribes audio files in "input" folder

import yaml
from pathlib import Path
from glob import glob
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from vtt_to_dense_vtt import vtt_to_dense_vtt


def read_config(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def transcribe_file(new_file):
    print('##### transcribing ' + new_file + ' ######')
        
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
        ], check=False
        )
    
    if process.returncode == 0:
        vtt_file = str(new_file)[:-4] + '.vtt'
        vtt_to_dense_vtt(vtt_file)
        log_entry = str(os.path.basename(new_file))
    else:
        log_entry = 'ERROR ' + str(os.path.basename(new_file))

    with open(Path(parent_dir, 'processed.txt'), 'a') as processed_file:    
        processed_file.write(log_entry)
        processed_file.write('\n')


def find_unprocessed_files(dir, extension_patterns):
    extensions = [p[1:] for p in extension_patterns]
    files_in_dir = glob(os.path.join(dir, '**'), recursive=True)
    audio_files =  [f for f in files_in_dir if os.path.splitext(f)[1] in extensions]
    new_audio_files = [f for f in audio_files if not os.path.isfile(os.path.splitext(f)[0] + '.vtt')]
    return new_audio_files


def on_new_file(event):
    if not event.is_directory:
        print(event.src_path)
        # transcribe_file(event.src_path)



if __name__ == '__main__':     
    cwd = os.path.abspath(os.path.dirname(__file__))
    global_cfg = read_config(Path(cwd, 'global-config.yml'))
    audio_extensions = ['*.mp3', '*.m4a', '*.flac', '*.mp4', '*.wav', '*.wma', '*.aac', '*.aiff', '*.pcm', '*.ogg', '*.vobis']

    # process new files
    unprocessed_files = find_unprocessed_files(global_cfg['input_dir'], audio_extensions)
    for f in unprocessed_files:
        transcribe_file(f)

    handler = PatternMatchingEventHandler(patterns = audio_extensions, case_sensitive = False)
    handler.on_created = on_new_file
    observer = Observer()
    observer.schedule(handler, global_cfg['input_dir'], recursive =True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        print("Observer stopping")
    finally:
        observer.stop()
        observer.join()