import csv
import argparse
from collections import OrderedDict
import random

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import pandas as pd
import numpy as np
import re


def get_pretrained_glove(path, idx2word, n_special=4):
    saved_glove = path.replace('.txt', '.pt')
    def make_glove():
        print('Reading pretrained glove...')
        words = pd.read_csv(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        def get_vec(w):
            # w = re.sub(r'\W+', '', w).lower() # SW: word tokenization needed
            w = w.lower()
            try:
                return words.loc[w].values.astype('float32')
            except KeyError: #_NAF_H, _NAF_R, _NAF_T
                return np.zeros((300,), dtype='float32')

        weights = [torch.from_numpy(get_vec(w)) for i, w in list(idx2word.items())[n_special:]]
        weights = torch.stack(weights, dim=0)

        addvec = torch.randn(n_special, weights.size(1))
        weights = torch.cat([addvec, weights], dim=0)
        torch.save(weights, saved_glove)
        print(f"Glove saved in {saved_glove}")
    if not os.path.isfile(saved_glove):
        make_glove()
    return torch.load(saved_glove)


class CCMModel(nn.Module):
    def __init__(self, args, idx2word):
        super().__init__()
        self.args = args
        print(len(idx2word), args.n_word_vocab)
        assert len(idx2word) == args.n_word_vocab, 'idx2word size does not match designated vocab size'
        # self.word_embedding = nn.Embedding(args.n_word_vocab, args.d_embed)
        self.word_embedding = nn.Embedding.from_pretrained(
            get_pretrained_glove(path=args.glove_path, idx2word=idx2word, n_special=4),
            freeze=False, padding_idx=0) # specials: pad, unk, naf_h/t
        self.transe_embedding = nn.Embedding(args.n_entity_vocab, args.t_embed)
        self.wh = nn.Linear(args.t_embed, args.hidden)
        self.wr = nn.Linear(args.t_embed, args.hidden)
        self.wt = nn.Linear(args.t_embed, args.hidden)
        self.gru_enc = nn.GRU(args.d_embed + 2 * args.t_embed, args.gru_hidden)
        self.gru_dec = nn.GRU(args.gru_hidden + 8 * args.t_embed + args.d_embed, args.gru_hidden)
        self.out = nn.Linear(args.gru_hidden + 8 * args.t_embed + args.d_embed, args.n_word_vocab)

    def forward(self, batch):
        post = batch['post']
        post_length = batch['post_length']
        response = batch['response']
        response_length = batch['response_length']
        post_triple = batch['post_triple']
        triple = batch['triple']
        triple_mask = triple.ne(0)
        entity = batch['entity']
        response_triple = batch['response_triple']

        post_emb = self.word_embedding(post)  # (bsz, pl, d_embed)
        bsz, rl = response.size()
        response_emb = self.word_embedding(response)  # (bsz, rl, d_embed)
        response_triple_emb = self.transe_embedding(response_triple) # (bsz, rl, 3, t_embed)
        response_triple_emb_flatten = torch.flatten(response_triple_emb, 2) # (bsz, rl, 3 * t_embed)
        triple_emb = self.transe_embedding(triple) # (bsz, pl, tl, 3, t_embed)
        triple_emb_flatten = torch.flatten(triple_emb, 3) # (bsz, pl, tl, 3 * t_embed)

        # TODO: Transform TransE

        # Static Graph
        head, rel, tail = torch.split(triple_emb, 1, 3)  # (bsz, pl, tl, t_embed)
        ent = torch.cat([head, tail], -1).squeeze(-2)  # (bsz, pl, tl, 2 * t_embed)
        static_logit = (self.wr(rel) * torch.tanh(self.wh(head) + self.wt(tail))).sum(-1).squeeze(-1)  # (bsz, pl, tl)
        if triple_mask is not None:
            static_logit.data.masked_fill_(triple_mask[:, :, :, 0], -float('inf'))
        static_attn = F.softmax(static_logit, dim=-1)  # (bsz, pl, tl)
        static_graph = (ent * static_attn.unsqueeze(-1)).sum(-2)  # (bsz, pl, 2 * t_embed) / gi
        post_triples = static_graph[post_triple]  # (bsz, pl, 2 * t_embed)
        post_input = torch.cat([post_emb, post_triples], -1)  # (bsz, pl, d_emb + 2 * t_embed)

        # Encoder
        packed_post_input = pack_padded_sequence(post_input, lengths=post_length.tolist(), batch_first=True)
        packed_post_output, gru_hidden = self.gru_enc(packed_post_input)
        post_output, gru_hidden = pad_packed_sequence(packed_post_output, batch_first=True)  # (bsz, pl, go)

        # Decoder
        response_input = torch.cat([response_emb, response_triple_emb_flatten], -1)  # (bsz, rl, d_embed + 3 * t_embed)
        dec_logits = torch.zeros(rl - 1, bsz, self.n_vocab).to(response.device)

        for t in range(rl - 1):
            response_vector = response_input[:, t]  # (bsz, d_embed + 3 * t_embed)
            #c
            context_logit = (post_output * gru_hidden.unsqueeze(-2)).sum(-1) # (bsz, pl)
            context_attn = F.softmax(context_logit, dim=-1)  # (bsz, pl)
            context_vector = (post_output * context_attn.unsqueeze(-1)).sum(-2)  # (bsz, gru_hidden) / c

            #cg
            dynamic_logit = self.vb(torch.tanh(self.wb(gru_hidden) + self.ub(static_graph))).sum(-1)  # (bsz, pl)
            dynamic_attn = F.softmax(dynamic_logit, dim=-1)  # (bsz, pl)
            dynamic_graph = (static_graph * dynamic_attn.unsqueeze(-1)).sum(-2)  # (bsz, 2 * t_embed) / cg

            #ck
            triple_logit = (triple_emb_flatten * self.wc(gru_hidden).unsqueeze(-2).unsqueeze(-2)).sum(-1) # (bsz, pl, tl)
            triple_attn = F.softmax(triple_logit, dim=-1) # (bsz, pl, tl)
            triple_vector = ((triple_emb_flatten * triple_attn.unsqueeze(-1)).sum(-2) * dynamic_attn.unsqueeze(-1)).sum(-1) # (bsz, 3 * t_embed)

            dec_input = torch.cat([context_vector, dynamic_graph, triple_vector, response_vector], 1).unsqueeze(-1) # (bsz, gru_hidden + 8 * t_embed + d_embed)
            gru_out, gru_hidden = self.gru_dec(dec_input, gru_hidden)  # (bsz, 1, gru_hidden) / (bsz, gru_hidden)

            logit = self.out(gru_out)  # (b, 1, n_vocab)
            dec_logits[t] = logit.transpose(0, 1)

        return dec_logits.permute(1, 2, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--glove_path', type=str, default='data/glove.840B.300d.txt')
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--t_embed', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_word_vocab', type=int, default=5441)
    parser.add_argument('--n_entity_vocab', type=int, default=22590)
    parser.add_argument('--gru_layer', type=int, default=2)
    parser.add_argument('--gru_hidden', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_sentence_len', type=int, default=150)
    parser.add_argument('--max_triple_len', type=int, default=50)
    parser.add_argument('--init_chunk_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    args.cuda = 'cpu'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.n_word_vocab += 4

    from dataloader import get_dataloader
    import pickle
    dataloader = get_dataloader(args=args, batch_size=2, shuffle=False)

    with open(f'{args.data_path}/vocab.pkl', 'rb') as vf:
        word2idx = pickle.load(vf)
    idx2word = {v:k for k, v in word2idx.items()}
    with open(f'{args.data_path}/resource.txt', 'rb') as vf:
        resource = eval(vf.read())
    for key, val in resource.items():
        print(key, len(val))
    # idx2word = OrderedDict(sorted(idx2word.items(), key=lambda t: t[0]))

    model = CCMModel(args, idx2word)
    model = model.to(args.cuda)

    batch = iter(dataloader).next()
    batch['post'] = torch.randint(high=args.n_word_vocab, size=batch['post'].size()).long()
    batch['response'] = torch.randint(high=args.n_word_vocab, size=batch['response'].size()).long()
    batch['post_triple'] = batch['post_triple'].long()
    batch['response_triple'] = batch['response_triple'].long()
    batch['triple'] = batch['triple'].long()
    batch = {key: val.to(args.cuda) for key, val in batch.items()}
    for key in batch.keys():
        print(key, batch[key].size())

    model(batch)
