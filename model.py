import csv
import argparse
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import pandas as pd
import numpy as np
import ipdb


def get_pretrained_glove(path, n_word=30004):
    saved_glove = path.replace('.txt', '.pt')
    def make_glove():
        print('Reading pretrained glove...')
        default = ['PAD', 'UNK', 'SOS', 'EOS']
        words = pd.read_csv(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        def get_vec(w):
            return words.loc[w].values.astype('float32')
        weights = [torch.from_numpy(get_vec(w)).unsqueeze(0) for i, w in enumerate(default)]
        weights.append(torch.from_numpy(words.iloc[:n_word - len(default), :].values.astype('float32')))
        weights = torch.cat(weights, dim=0)

        torch.save(weights, saved_glove)
        print(f"Glove saved in {saved_glove}")
    if not os.path.isfile(saved_glove):
        make_glove()
    return torch.load(saved_glove)


def get_pretrained(label_path, weight_path, idx2word, dim=100):
    saved_weight = weight_path.replace('.txt', '.pt')
    def make_weights():
        labels = [label for label in open(label_path, 'r').read().split('\n') if label]
        entity = pd.read_csv(weight_path, sep="\t", header=None, quoting=csv.QUOTE_NONE)
        entity.index = labels
        n = 0
        def get_vec(w):
            nonlocal n
            try:
                return entity.loc[w].values.astype('float32')[:dim]
            except KeyError: #_NAF_H, _NAF_R, _NAF_T
                print(w)
                n += 1
                return np.zeros(dim).astype('float32')

        weights = [torch.from_numpy(get_vec(w)).unsqueeze(0) for i, w in idx2word.items()]
        print(n, len(idx2word))
        weights = torch.cat(weights, dim=0)
        torch.save(weights, saved_weight)
        print(f"Weights saved in {saved_weight}")
    if not os.path.isfile(saved_weight):
        make_weights()
    return torch.load(saved_weight)


def get_pad_mask(lengths, max_length):
    """ 1 for pad """
    bsz = lengths.size()[0]
    mask = torch.zeros((bsz, max_length), dtype=torch.bool)
    for j in range(bsz):
        mask[j, lengths[j]+1:] = 1
    return mask


class CCMModel(nn.Module):
    def __init__(self, args, idx2ent, idx2rel):
        super().__init__()
        self.args = args
        self.idx2ent = idx2ent
        self.idx2rel = idx2rel
        self.n_word_vocab = args.n_word_vocab
        self.gru_layer = args.gru_layer

        self.word_embedding = nn.Embedding.from_pretrained(
            get_pretrained_glove(path=f'{args.data_dir}/glove.840B.300d.txt', n_word=args.n_word_vocab),
            freeze=False, padding_idx=0) # specials: pad, unk, naf_h/t

        self.entity_embedding = nn.Embedding.from_pretrained(
            get_pretrained(label_path=f'{args.data_dir}/entity.txt', weight_path=f'{args.data_dir}/entity_transE.txt', idx2word=idx2ent),
            freeze=False, padding_idx=0)

        self.rel_embedding = nn.Embedding.from_pretrained(
            get_pretrained(label_path=f'{args.data_dir}/relation.txt', weight_path=f'{args.data_dir}/relation_transE.txt', idx2word=idx2rel),
            freeze=False, padding_idx=0)

        self.wh = nn.Linear(args.t_embed, args.hidden)
        self.wr = nn.Linear(args.t_embed, args.hidden)
        self.wt = nn.Linear(args.t_embed, args.hidden)
        self.gru_enc = nn.GRU(args.d_embed + 2 * args.t_embed, args.gru_hidden, args.gru_layer, batch_first=True)
        self.gru_dec = nn.GRU(args.gru_hidden + 8 * args.t_embed + args.d_embed, args.gru_hidden, args.gru_layer, batch_first=True)
        self.wa = nn.Linear(args.gru_hidden * args.gru_layer, args.gru_hidden)
        self.wb = nn.Linear(args.gru_hidden * args.gru_layer, args.hidden)
        self.ub = nn.Linear(2 * args.t_embed, args.hidden)
        self.vb = nn.Linear(args.hidden, 1)
        self.wc = nn.Linear(args.gru_hidden * args.gru_layer, 3 * args.t_embed)
        self.out = nn.Linear(args.gru_hidden, args.n_word_vocab)

    def forward(self, batch):
        post = batch['post']
        post_mask = post.eq(0)
        post_length = batch['post_length']
        response = batch['response']
        response_length = batch['response_length']
        post_triple = batch['post_triple']
        triple = batch['triple']
        triple_mask = triple.eq(0)
        entity = batch['entity']
        response_triple = batch['response_triple']
        device = post.device

        bsz, rl = response.size()
        post_emb = self.word_embedding(post)  # (bsz, pl, d_embed)
        response_emb = self.word_embedding(response)  # (bsz, rl, d_embed)
        head, rel, tail = torch.split(triple, 1, 3)  # (bsz, pl, tl)
        head_emb = self.entity_embedding(head.squeeze(-1))  # (bsz, pl, tl, t_embed)
        rel_emb = self.rel_embedding(rel.squeeze(-1)) # (bsz, pl, tl, t_embed)
        tail_emb = self.entity_embedding(tail.squeeze(-1))  # (bsz, pl, tl, t_embed)
        triple_emb = torch.cat([head_emb, rel_emb, tail_emb], 3)  # (bsz, pl, tl, 3 * t_embed)
        res_head, res_rel, res_tail = torch.split(response_triple, 1, 2)  # (bsz, rl, 1)
        res_head_emb = self.entity_embedding(res_head.squeeze(-1))  # (bsz, rl, t_embed)
        res_rel_emb = self.rel_embedding(res_rel.squeeze(-1))  # (bsz, rl, t_embed)
        res_tail_emb = self.entity_embedding(res_tail.squeeze(-1))  # (bsz, rl, t_embed)
        res_triple_emb = torch.cat([res_head_emb, res_rel_emb, res_tail_emb], 2)  # (bsz, rl, 3 * t_embed)
        # TODO: Transform TransE

        # Static Graph
        ent = torch.cat([head_emb, tail_emb], -1)  # (bsz, pl, tl, 2 * t_embed)
        mask = get_pad_mask(post_triple.max(1)[0], ent.size()[1]).to(device)
        ent.data.masked_fill_(mask.unsqueeze(-1).unsqueeze(-1), 0)
        static_logit = (self.wr(rel_emb) * torch.tanh(self.wh(head_emb) + self.wt(tail_emb))).sum(-1, keepdim=False)  # (bsz, pl, tl)
        static_logit.data.masked_fill_(triple_mask[:, :, :, 0], -float('inf'))
        static_logit.data.masked_fill_(mask.unsqueeze(-1), 0)
        static_attn = F.softmax(static_logit, dim=-1)  # (bsz, pl, tl)
        static_graph = (ent * static_attn.unsqueeze(-1)).sum(-2)  # (bsz, pl, 2 * t_embed) / gi
        post_triples = static_graph.gather(1, post_triple.unsqueeze(-1).expand(-1, -1, static_graph.size()[-1]))
        post_input = torch.cat([post_emb, post_triples], -1)  # (bsz, pl, d_emb + 2 * t_embed)

        # Encoder
        packed_post_input = pack_padded_sequence(post_input, lengths=post_length.tolist(), batch_first=True)
        packed_post_output, gru_hidden = self.gru_enc(packed_post_input)
        post_output, _ = pad_packed_sequence(packed_post_output, batch_first=True)  # (bsz, pl, go)
        gru_state = gru_hidden.transpose(0, 1).reshape(bsz, 1, -1)

        # Decoder
        response_input = torch.cat([response_emb, res_triple_emb], -1)  # (bsz, rl, d_embed + 3 * t_embed)
        dec_logits = torch.zeros(rl - 1, bsz, self.n_word_vocab).to(device)

        for t in range(rl - 1):
            response_vector = response_input[:, t]  # (bsz, d_embed + 3 * t_embed)
            #c
            context_logit = (post_output * self.wa(gru_state)).sum(-1) # (bsz, pl)
            context_logit.data.masked_fill_(post_mask, -float('inf'))
            context_attn = F.softmax(context_logit, dim=-1)  # (bsz, pl)
            context_vector = (post_output * context_attn.unsqueeze(-1)).sum(-2, keepdim=False)  # (bsz, gru_hidden) / c

            #cg
            dynamic_logit = self.vb(torch.tanh(self.wb(gru_state) + self.ub(static_graph))).squeeze(-1)  # (bsz, pl)
            dynamic_logit.data.masked_fill_(mask, -float('inf'))
            dynamic_attn = F.softmax(dynamic_logit, dim=-1)  # (bsz, pl)
            dynamic_graph = (static_graph * dynamic_attn.unsqueeze(-1)).sum(-2)  # (bsz, 2 * t_embed) / cg

            #ck
            triple_logit = (triple_emb * self.wc(gru_state).unsqueeze(-2)).sum(-1) # (bsz, pl, tl)
            triple_logit.data.masked_fill_(triple_mask[:, :, :, 0], -float('inf'))
            triple_logit.data.masked_fill_(mask.unsqueeze(-1), 0)
            triple_attn = F.softmax(triple_logit, dim=-1) # (bsz, pl, tl)
            triple_tmp = (triple_emb * triple_attn.unsqueeze(-1)).sum(-2, keepdim=False)
            triple_tmp.data.masked_fill_(mask.unsqueeze(-1), 0)
            triple_vector = (triple_tmp * dynamic_attn.unsqueeze(-1)).sum(-2) # (bsz, 3 * t_embed)

            dec_input = torch.cat([context_vector, dynamic_graph, triple_vector, response_vector], 1).unsqueeze(-2) # (bsz, gru_hidden + 8 * t_embed + d_embed)
            gru_out, gru_hidden = self.gru_dec(dec_input, gru_hidden)  # (bsz, 1, gru_hidden) / (bsz, gru_hidden)
            gru_state = gru_hidden.transpose(0, 1).reshape(bsz, 1, -1)

            logit = self.out(gru_out)  # (bsz, 1, n_vocab)
            dec_logits[t] = logit.transpose(0, 1)

        return dec_logits.permute(1, 2, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--t_embed', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_word_vocab', type=int, default=30000)
    parser.add_argument('--n_entity_vocab', type=int, default=22590)
    parser.add_argument('--gru_layer', type=int, default=2)
    parser.add_argument('--gru_hidden', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_sentence_len', type=int, default=150)
    parser.add_argument('--max_triple_len', type=int, default=50)
    parser.add_argument('--init_chunk_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.n_word_vocab += 4

    from dataloader import get_dataloader
    dataloader = get_dataloader(args=args, batch_size=args.batch_size, shuffle=False)

    model = CCMModel(args, dataloader.dataset.idx2ent, dataloader.dataset.idx2rel)
    batch = iter(dataloader).next()
    model(batch)