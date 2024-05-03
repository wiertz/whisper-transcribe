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
        content_split = re.split(r'(?<=\]): ', content)
        if len(content_split) > 1:
            speaker = content_split[0]
            text = ': '.join(content_split[1:])
        else:
            speaker = None
            text = content_split[0]
        
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

    current_block = None
    new_blocks = []

    for block in vtt:
        parsed_block = parse_block(block)
        
        # reached end or inconsistency in file:
        if not parsed_block:
            if current_block: 
                new_blocks.append(current_block.copy())
                current_block = None
            continue
               
        # no current block (first block)
        if not current_block:
            current_block = parsed_block.copy()
            continue
        
        # if speaker missing, use previous speaker
        if not parsed_block['speaker']:
            parsed_block['speaker'] = current_block['speaker']
        
        # extend current block if speaker continues and time limit not exceeded
        if (parsed_block['speaker'] == current_block['speaker']) and (current_block['duration'] <= 30):
            current_block['end'] = parsed_block['end']
            current_block['text'] += ' ' + parsed_block['text']
            current_block['duration'] += parsed_block['duration']
            continue
        
        # start new block with parsed block
        new_blocks.append(current_block.copy())
        current_block = parsed_block.copy()

    with open(out_file, 'w') as out:
        out.write(format_blocks(new_blocks))