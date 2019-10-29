# CCM-pytorch
Pytorch implementation of 'Commonsense Knowledge Aware Conversation Generation with Graph Attention'



### Preparation

1. Download data from [here](http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense) and unzip at 'data' folder

2. Change file extension of {train, valid, test}.txt to .jsonl

3. Divide jsonl files into smaller files under '{train, valid, test}set_pieces' folder

   e.g. `split -l 10000 trainset.jsonl trainset_pieces/piece_`, and set: args.init_chunk_size = 10000

4. Replace 'glove.840B.300d.txt' under the 'data' folder with the [real file](https://nlp.stanford.edu/projects/glove/) holding pretrained weights

5. `pip install -r requirements.txt`

