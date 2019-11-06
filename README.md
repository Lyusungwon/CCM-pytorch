# CCM-pytorch
Pytorch implementation of 'Commonsense Knowledge Aware Conversation Generation with Graph Attention'



### Preparation

1. Download data from [here](http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense) and unzip at 'data' folder

2. Change file extension of {train, valid, test}.txt to .jsonl

3. Divide jsonl files into smaller files under '{train, valid, test}set_pieces' folder

   e.g. `split -l 10000 trainset.jsonl trainset_pieces/piece_`, and set: args.init_chunk_size = 10000

4. Replace 'glove.840B.300d.txt' under the 'data' folder with the [real file](https://nlp.stanford.edu/projects/glove/) holding pretrained weights

5. `pip install -r requirements.txt`



### Storing ConceptNet triples with RedisGraph

1. Install and build [Redis](https://redis.io) and [RedisGraph](https://oss.redislabs.com/redisgraph/)

2. Open redis-server and load RedisGraph module:

   `redis-server --loadmodule /path/to/module/src/redisgraph.so`

3. `python graph.py` will store triples on your RAM

4. After all triples are stored, `redis-cli bgsave`

5. [Making AOF] For safety, make a backup of your latest dump.rdb file and transfer this backup to a safe place; then `redis-cli config set appendonly yes; redis-cli config set save ""`

