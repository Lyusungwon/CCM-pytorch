import os
from ast import literal_eval
from tqdm import tqdm
import jsonlines
import redis
from fuzzywuzzy import process, fuzz

import ipdb

def store_graph(rd, triples):
    print('Storing triples as RedisGraph...')
    for triple in tqdm(triples):
        head, rel, tail = triple.split(", ")
        rd.execute_command('GRAPH.QUERY', 'CCM', f"MERGE ({{word: '{head}'}})")
        rd.execute_command('GRAPH.QUERY', 'CCM', f"MERGE ({{word: '{tail}'}})")
        rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x), (y) WHERE x.word = '{head}' AND y.word = '{tail}' CREATE (x)-[:{rel}]->(y)")


def retrieve_graph(rd, query, query_as_head=True, fuzzy=False, entity_lst=None):
    if fuzzy and entity_lst is not None:
        # Fuzzy string match
        query = process.extractOne(query, entity_lst, scorer=fuzz.token_sort_ratio)[0]
    print(query)
    if query_as_head:
        resp = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x)-[r]->(y) WHERE x.word = '{query}' RETURN r, y.word")
    else:
        resp = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x)-[r]->(y) WHERE y.word = '{query}' RETURN r, x.word")
    return [(rel[1][1].decode('utf-8'), ent.decode('utf-8')) for rel, ent in resp[1]]
    

if __name__ == '__main__':
    rd = redis.StrictRedis()

    raw_dict = open('data/resource.txt', 'r').read()
    raw_dict = literal_eval(raw_dict)

    # STORE
    store_graph(rd, raw_dict['csk_triples'])

    # QUERY
    # print(retrieve_graph(rd, 'fawn', fuzzy=True, entity_lst=raw_dict['csk_entities']))
    # print(retrieve_graph(rd, 'faun', fuzzy=True, entity_lst=raw_dict['csk_entities']))
