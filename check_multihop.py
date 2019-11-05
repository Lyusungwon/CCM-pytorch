import os
from ast import literal_eval
import redis

rd = redis.StrictRedis()

raw_dict = open('data/resource.txt', 'r').read()
raw_dict = literal_eval(raw_dict)
entity_lst = raw_dict['csk_entities']

all_cnt = 0
onehop_cnt, twohop_cnt, threehop_cnt = 0, 0, 0

with jsonlines.open('data/trainset.jsonl', mode='r') as reader:
    for i, line in enumerate(reader):
        if i > 10000: break

        pairs = [(p, r) for p in line['post'] for r in line['response'] if p in entity_lst and r in entity_lst]
        all_cnt += len(pairs)
        for p, r in pairs:
            # 1-hop
            reply = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x)-[]->(y) WHERE x.word = '{p}' AND y.word = '{r}' RETURN y.word")
            if len(reply[1]) > 0:
                onehop_cnt += 1

            # 1~2
            reply = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x)-[*1..2]->(y) WHERE x.word = '{p}' AND y.word = '{r}' RETURN y.word")
            if len(reply[1]) > 0:
                twohop_cnt += 1
            
            # 1~3
            reply = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x)-[*1..3]->(y) WHERE x.word = '{p}' AND y.word = '{r}' RETURN y.word")
            if len(reply[1]) > 0:
                threehop_cnt += 1

        print(i, ' : ', all_cnt, onehop_cnt, twohop_cnt, threehop_cnt)

print(all_cnt, onehop_cnt, twohop_cnt, threehop_cnt)