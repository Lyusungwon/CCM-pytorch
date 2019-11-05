import os
from ast import literal_eval
from tqdm import tqdm
import jsonlines
import redis

rd = redis.StrictRedis()
# rd.execute_command('GRAPH.DELETE', 'CCM')

raw_dict = open('data/resource.txt', 'r').read()
raw_dict = literal_eval(raw_dict)

for triple in tqdm(raw_dict['csk_triples']):
    head, rel, tail = triple.split(", ")
    rd.execute_command('GRAPH.QUERY', 'CCM', f"MERGE ({{word: '{head}'}})")
    rd.execute_command('GRAPH.QUERY', 'CCM', f"MERGE ({{word: '{tail}'}})")
    resp = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x), (y) WHERE x.word = '{head}' AND y.word = '{tail}' CREATE (x)-[:{rel}]->(y)")