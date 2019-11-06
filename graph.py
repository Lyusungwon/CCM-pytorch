import os
from ast import literal_eval
from tqdm import tqdm
import jsonlines
import redis
from fuzzywuzzy import process, fuzz

def store_graph(rd, triples):
    for triple in tqdm(triples):
        head, rel, tail = triple.split(", ")
        rd.execute_command('GRAPH.QUERY', 'CCM', f"MERGE ({{word: '{head}'}})")
        rd.execute_command('GRAPH.QUERY', 'CCM', f"MERGE ({{word: '{tail}'}})")
        rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x), (y) WHERE x.word = '{head}' AND y.word = '{tail}' CREATE (x)-[:{rel}]->(y)")


def retrieve_graph(rd, query, entity_lst):
    """ Fuzzy string match """
    query = process.extractOne(query, entity_lst, scorer=fuzz.token_sort_ratio)[0]
    print(query)
    resp = rd.execute_command('GRAPH.QUERY', 'CCM', f"MATCH (x)-[]->(y) WHERE x.word = '{query}' RETURN y.word")
    return [el[0].decode('utf-8') for el in resp[1]]
    

if __name__ == '__main__':
    rd = redis.StrictRedis()

    raw_dict = open('data/resource.txt', 'r').read()
    raw_dict = literal_eval(raw_dict)

    # STORE
    # store_graph(rd, raw_dict['csk_triples'])

    # QUERY
    print(retrieve_graph(rd, 'fawn', raw_dict['csk_entities']))
    print(retrieve_graph(rd, 'faun', raw_dict['csk_entities']))
