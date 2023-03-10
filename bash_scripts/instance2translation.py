import os
import sys
import json
from sacremoses import MosesTokenizer, MosesDetokenizer

def read_file(f):
    results = []
    with open(f, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip('\n')
            results.append(l)
            # if len(results) > 10:
                # break
    return results

def write_file(file, lines):
    with open(file, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l+'\n')

def get_detokenize(lang):  # get  detokenizer
    if lang not in ['zh', 'ja', 'ko']:
        moses_detoknizer = MosesDetokenizer(lang=lang)
        return lambda x: moses_detoknizer.detokenize(x.split())

if __name__ == '__main__':
    file = sys.argv[1]
    out_file = sys.argv[2]
    src_file = sys.argv[3]
    ref_file = sys.argv[4]
    lines = read_file(file)
    # en_detoknizer = get_detokenize('en')

    pred_lines = []
    src_lines = []
    ref_lines = []
    for l in lines:
        # print(l)
        d = json.loads(l)
        pred = d['prediction']
        pred = ' '.join(pred.split()[:-1])
        # pred = en_detoknizer(pred)
        # print(pred)
        pred_lines.append(pred)
        src_lines.append(d['source'])
        ref_lines.append(d['reference'])

    assert len(src_lines) == len(ref_lines) == len(pred_lines)

    write_file(out_file, pred_lines)

    if src_file:
        write_file(src_file, src_lines)

    if ref_file:
        write_file(ref_file, ref_lines)
