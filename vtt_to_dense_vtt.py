import re
from datetime import datetime
   
def parse_timestamp(ts):
    n_segments = len(re.findall(':', ts))
    if(n_segments == 2):
        t = datetime.strptime(ts, '%H:%M:%S.%f')
    else:
        t = datetime.strptime(ts, '%M:%S.%f')
    return t


def format_timestamp(ts):
    new_ts = datetime.strftime(ts, '%H:%M:%S.%f')[:-3]
    return new_ts


def parse_block(block):
    try:
        timestamp, content = block.split('\n')
        begin_str, end_str = timestamp.split(' --> ')
        begin = parse_timestamp(begin_str)
        end = parse_timestamp(end_str)
        speaker, text = re.split(r'(?<=\]): ', content)
        
        return({
            'begin': begin,
            'end': end,
            'duration': (end - begin).total_seconds(),
            'speaker': speaker,
            'text': text
            })
    except Exception:
        return None


def format_blocks(blocks):
    txt = ['WEBVTT']
    for b in blocks:
        begin, end, duration, speaker, text = b.values()
        block_txt = f'{format_timestamp(begin)} --> {format_timestamp(end)}\n{speaker}: {text}'
        txt.append(block_txt)
    return '\n\n'.join(txt)
    



def vtt_to_dense_vtt(in_file):
    out_file = in_file[:-4] + '.dense.vtt'
    
    with open(in_file) as f:
        vtt = f.read().split('\n\n')[1:]

    current_speaker = None
    new_blocks = []
    current_block = {}

    for block in vtt:
        parsed_block = parse_block(block)
        if parsed_block is None:
            if len(current_block) > 0:
                new_blocks.append(current_block)
                current_block = {}
            continue

        begin, end, duration, speaker, text = parsed_block.values()
        
        if (speaker != current_speaker) or (current_block['duration'] > 30):
            if current_speaker is not None:
                new_blocks.append(current_block)
            
            current_speaker = speaker
            current_block = parsed_block

        else:
            current_block['end'] = end
            current_block['text'] += ' ' + text
            current_block['duration'] += duration
            

    with open(out_file, 'w') as out:
        out.write(format_blocks(new_blocks))