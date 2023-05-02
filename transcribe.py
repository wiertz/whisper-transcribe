# Strongly inspired by https://colab.research.google.com/github/Majdoddin/nlp/blob/main/Pyannote_plays_and_Whisper_rhymes_v_2_0.ipynb

import os
import math
import sys
import subprocess
from pathlib import Path
from shutil import rmtree
from time import strftime, gmtime

from pydub import AudioSegment
import torch
import whisper


# set parameters
input_dir = Path("input")
output_dir = Path("output")
processed_dir = Path("processed")
ffmpeg_path = Path('/usr/local/bin/ffmpeg')
offset_ms = 2000    # offset to add to the start of the audio file for better diarization (necessary?)
# ffmpet_dir = Path('/usr/bin/ffmpeg')
hf_token = "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"
model = "large-v2"



def append_audio(audio_file, temp_dir, offset_ms):
    audio_file_in = str(audio_file.resolve())
    audio_file_prep = str(Path(temp_dir, audio_file.name).with_suffix(".prep.wav").resolve())
    subprocess.run(
        [ffmpeg_path, '-i', audio_file_in, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audio_file_prep], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL, 
        check=True
        )
    spacer = AudioSegment.silent(duration=offset_ms)
    audio = AudioSegment.from_wav(audio_file_prep)
    audio = spacer.append(audio, crossfade=0)
    audio.export(audio_file_prep, format='wav')
    return Path(audio_file_prep)


def diarize(audio_file):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    FILE = {'uri': 'none', 'audio': str(audio_file)}
    diarization = pipeline(FILE)
    return [{"start": d[0].start, "end": d[0].end, "speaker": d[-1]} for d in diarization.itertracks(yield_label = True)]


def group_segments(diarization_result):
    groups = []
    current_start = None
    current_end = None
    current_speaker = None

    for d in diarization_result:
        if not current_speaker:
            current_speaker = d['speaker']
            current_start = d['start']
            current_end = d['end']

        elif current_speaker and d['speaker'] == current_speaker:
            current_end = d['end']

        else:
            groups.append({'start': current_start, 'end': current_end, 'speaker': current_speaker})
            current_speaker = d['speaker']
            current_start = d['start']
            current_end = d['end']

    return groups


def split_audio(audio_file, groups, temp_dir):
    audio = AudioSegment.from_wav(audio_file)

    split_files = []
    for idx, g in enumerate(groups):
        start_ms = math.floor(g['start'] * 1000)
        end_ms = math.ceil(g['end'] * 1000)
        out_file = str(Path(temp_dir, str(idx) + '.wav'))
        audio[start_ms:end_ms].export(out_file, format='wav')
        split_files.append(out_file)

    return split_files


def transcribe_files(split_files, model, language):
    device = torch.device('cpu' if model in ['large', 'large-v2'] or not torch.cuda.is_available() else 'cuda')
    model = whisper.load_model(model, device)
    transcript = []

    for f in split_files:
        t = model.transcribe(audio=f, language=language, word_timestamps=False)
        print(t['text'])
        transcript.append(t['text'])

    return transcript

def timestamp_from_sec(time_in_seconds):
    time_hms = strftime('%H:%M:%S', gmtime(time_in_seconds))
    time_ds = str(round(time_in_seconds % 1, 1)).split('.')[-1]
    return f'{time_hms}.{time_ds}'

def transcribe(audio_file, model, language, out_file, temp_dir, offset_ms):
    audio_file_prep = append_audio(audio_file, temp_dir, offset_ms)
    diarization = diarize(audio_file_prep)
    segment_groups = group_segments(diarization)
    split_files = split_audio(audio_file_prep, segment_groups, temp_dir)
    transcript = transcribe_files(split_files, model=model, language=language)

    with open(out_file, 'w') as f:
        transcript_with_info = zip(segment_groups, transcript)
        for segment in transcript_with_info:
            time_stamp_end = timestamp_from_sec(segment[0]['end'] - offset_ms / 1000)
            f.write(f"{segment[0]['speaker']}:{segment[1]}\n{time_stamp_end}\n")


if __name__ == '__main__':
    arguments = sys.argv
    if len(sys.argv) != 2 or len(sys.argv[1]) != 2:
        raise ValueError('provide language as parameter (e.g. "python transcribe.py de")')

    language = sys.argv[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [f for f in input_dir.glob('*') if Path.is_file(f)]

    print(f'\nfound {len(audio_files)} files to transcribe...')

    for file in audio_files:
        print(f'beginning transcription of {file}... (language: {language})\n')
        try:
            temp_dir = Path('temp', file.name)
            temp_dir.mkdir(parents=True, exist_ok=True)
            out_file = Path('output', file.name).with_suffix('.txt')
            transcribe(file, model, language, out_file, temp_dir, offset_ms)
            print(f'\bfinished transcription of {file}... (language: {language})\n')
        finally:
            # file.rename(processed_dir / file.name)
            rmtree(temp_dir)

