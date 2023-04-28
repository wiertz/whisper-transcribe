# Strongly inspired by https://colab.research.google.com/github/Majdoddin/nlp/blob/main/Pyannote_plays_and_Whisper_rhymes_v_2_0.ipynb

import os
import math
from pathlib import Path
from shutil import rmtree
from time import strftime, gmtime

from pydub import AudioSegment
import torch
import whisper


# set parameters
input_dir = Path("input")
output_dir = Path("output")
hf_token = "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"
model = "large"
language = "de"


def append_audio(audio_file, temp_dir):
    audio_file_in = str(audio_file.resolve())
    audio_file_prep = str(Path(temp_dir, audio_file.name).with_suffix(".prep.wav").resolve())
    os.system(f'/usr/local/bin/ffmpeg -i {repr(audio_file_in)} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y {repr(audio_file_prep)}')
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
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
        start_ms = math.floor(g['start'] * 1000) - 50
        end_ms = math.ceil(g['end'] * 1000) + 50
        out_file = str(Path(temp_dir, str(idx) + '.wav'))
        audio[start_ms:end_ms].export(out_file, format='wav')
        split_files.append(out_file)

    return split_files


def transcribe_files(split_files, model, language):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model, device)
    transcript = []

    for f in split_files:
        t = model.transcribe(audio=f, language=language, word_timestamps=False)
        print(t['text'])
        transcript.append(t['text'])

    return transcript


def transcribe(audio_file, model, language, out_file, temp_dir):
    audio_file_prep = append_audio(audio_file, temp_dir)
    diarization = diarize(audio_file_prep)
    segment_groups = group_segments(diarization)
    split_files = split_audio(audio_file_prep, segment_groups, temp_dir)
    transcript = transcribe_files(split_files, model=model, language=language)

    with open(out_file, 'w') as f:
        transcript_with_info = zip(segment_groups, transcript)
        for segment in transcript_with_info:
            start_time = strftime('%H:%M:%S', gmtime(segment[0]['start']))
            end_time = strftime('%H:%M:%S', gmtime(segment[0]['end']))
            f.write(f"{segment[0]['speaker']} ({start_time}): {segment[1]}\n")


if __name__ == '__main__':
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [f for f in input_dir.glob('*') if Path.is_file(f)]
    for file in audio_files:
        temp_dir = Path('temp', file.name)
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_file = Path('output', file.name).with_suffix('.txt')
        transcribe(file, model, language)
        temp_dir.r

