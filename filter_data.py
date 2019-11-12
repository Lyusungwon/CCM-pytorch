import os
from ast import literal_eval
from tqdm import tqdm
from collections import Counter, OrderedDict
import jsonlines
import redis
import pickle
import numpy as np

from utils import line_count

import ipdb

rd = redis.StrictRedis()

raw_dict = open('data/resource.txt', 'r').read()
raw_dict = literal_eval(raw_dict)
entity_lst = raw_dict['csk_entities']

if not os.path.isfile('freq_dict.pkl'):
    freq_dict = Counter()

    num_lines = line_count('data/trainset.jsonl')
    with jsonlines.open('data/trainset.jsonl', mode='r') as reader:
        for _, line in zip(tqdm(range(num_lines)), reader):
            freq_dict.update(line['post'])
            freq_dict.update(line['response'])

    with open('freq_dict.pkl', 'wb') as f:
        pickle.dump(freq_dict, f)


with open('freq_dict.pkl', 'rb') as f:
    freq_dict = pickle.load(f)

def is_in_khop(k_exp):
    reply = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x)-[{k_exp}]->(y) WHERE x.word = '{p}' AND y.word = '{r}' RETURN y.word")
    return len(reply[1]) > 0

with jsonlines.open('data/trainset.jsonl', mode='r') as reader:
    with open('gt_new_hopinfo.txt', 'w') as writer:
        for i, line in enumerate(reader):
            print(f'sample {i}')
            writer.write(f'\nsample {i}\n')
            writer.write('\t' + ' '.join(line['post']) + '\n')
            writer.write('\t' + ' '.join(line['response']) + '\n')

            pairs = OrderedDict((r, [p for p in line['post'] if p in entity_lst]) for r in line['response'] if r in entity_lst)
            for r in pairs:
                tmp, p = [], None
                for k, k_exp in enumerate(['*1', '*1..2', '*1..3'], start=1):
                    for p in pairs[r]:
                        if is_in_khop(k_exp):
                            tmp.append(p)
                    if len(tmp) == 0: continue
                    cnts = np.array([freq_dict[p] for p in tmp])
                    p = tmp[np.argmin(cnts)]
                    break
                print(f'\t\t{p}--{r} ({k})')
                writer.write(f'\t\t{p}--{r} ({k})')