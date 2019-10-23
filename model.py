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

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

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
                return np.zeros((200,), dtype='float32')

        weights = [torch.from_numpy(get_vec(w)) for i, w in list(idx2word.items())[n_special:]]
        weights = torch.stack(weights, dim=0)

        addvec = torch.randn(n_special, weights.size(1))
        weights = torch.cat([addvec, weights], dim=0)
        torch.save(weights, saved_glove)
        print(f"Glove saved in {saved_glove}")
    if not os.path.isfile(saved_glove):
        make_glove()
    return torch.load(saved_glove)


def get_pad_mask(valid_bsz_by_timestep):
    """ 1 for pad """
    bsz, t = valid_bsz_by_timestep[0], len(valid_bsz_by_timestep)
    mask = torch.zeros((bsz, t), dtype=torch.bool)
    for j in range(1, t):
        mask[valid_bsz_by_timestep[j]:, j] = 1
    return mask


class IEMSAModel(nn.Module):
    def __init__(self, args, idx2word):
        super().__init__()
        self.args = args
        assert len(idx2word) == args.n_word_vocab, 'idx2word size does not match designated vocab size'
        self.n_vocab = args.n_word_vocab

        self.word_embedding = nn.Embedding.from_pretrained(
            get_pretrained_glove(path=args.glove_path, idx2word=idx2word, n_special=4),
            freeze=False, padding_idx=0) # specials: pad, unk, naf_h/t
        self.rel_embedding = nn.Embedding(args.n_rel_vocab, args.d_embed) # naf_r 빠져야.

        # shared by all post encoders and final decoder
        self.lstm = nn.LSTM(input_size=args.d_embed, hidden_size=args.d_hidden, num_layers=args.n_layer, batch_first=True)

        # For building graph vector
        self.graph_attn = GraphAttention(d_embed=args.d_embed, d_proj=args.d_hidden)

        self.attn = MultiSourceAttention(dim=args.d_hidden)

        self.out = nn.Linear(args.d_hidden, self.n_vocab) # logit

        self.init_weights()

    def init_weights(self):
        # TODO: init lstm hidden
        pass

    def forward(self, batch, teacher_force_ratio=0.5):
        # NOTE: 앞뒤로 sos(2), eos(3) 붙어오는 것 가정
        # mask: (bsz, len, n_triple)

        # SW: Would it be better to do this on dataloader?
        post_lst = [batch['post_1'], batch['post_2'], batch['post_3'], batch['post_4']] # (bsz, timestep)
        post_length_lst = [batch['post_length_1'], batch['post_length_2'], batch['post_length_3'], batch['post_length_4']] # (bsz)
        entity_lst = [batch['entity_1'], batch['entity_2'], batch['entity_3'], batch['entity_4']] # (bsz, timestep, n_triple, 3)
        entity_length_lst = [batch['entity_length_1'], batch['entity_length_2'], batch['entity_length_3'], batch['entity_length_4']] # (bsz, timestep)
        entity_mask_lst = [batch['entity_mask_1'], batch['entity_mask_2'], batch['entity_mask_3'], batch['entity_mask_4']] # (bsz, timestep, n_triple)
        response = batch['response'] # (bsz, timestep)
        ###

        # prev sentence
        cached_post, cached_post_mask = None, None
        cached_graph_vec, cached_graph_vec_mask = None, None
        lstm_hidden = None

        # supervision for encoding
        enc_logits = [] # for post2~4

        ### Encode all posts
        for i in range(4):
            post, post_len, ent, ent_len, ent_mask = post_lst[i], post_length_lst[i], entity_lst[i], entity_length_lst[i], entity_mask_lst[i]

            post = self.word_embedding(post) # (b, l, d_embed)

            head = self.word_embedding(ent[:, :, :, 0]).unsqueeze(-2)
            rel = self.rel_embedding(ent[:, :, :, 1]).unsqueeze(-2)
            tail = self.word_embedding(ent[:, :, :, 2]).unsqueeze(-2)
            ent = torch.cat([head, rel, tail], dim=3) # (b, l, n_triple, 3, d_embed)

            ### Encode post sequence.

            # Sort post sequence by descending length order and pack
            post_len, perm = post_len.sort(dim=0, descending=True)
            post = post[perm]
            packed_post = pack_padded_sequence(post, lengths=post_len.tolist(), batch_first=True)

            # make mask for sequence attention
            valid_bsz = packed_post.batch_sizes # Effective batch size at each timestep
            post_mask = get_pad_mask(valid_bsz).unsqueeze(1).to(post.device) # (b, l, l_q)
            
            # lstm-encode
            packed_post, lstm_hidden = self.lstm(packed_post, lstm_hidden)

            # restore post order
            post, _ = pad_packed_sequence(packed_post, batch_first=True) # restore by padding; (b, l_q, d)
            _, unperm = perm.sort(dim=0, descending=False)
            post = post[unperm]

            ### Make graph vector.
            graph_vec = self.graph_attn(ent, mask=ent_mask)
            graph_vec_mask = (ent_len == 0).unsqueeze(1) # (b, 1, l_kv); 1 for zero triples
            
            if i == 0:
                enc_output = post
            else:
                # attention on previous post
                enc_output, *_ = self.attn(query=post, state_keyval=cached_post, knowledge_keyval=cached_graph_vec, state_mask=cached_post_mask, knowledge_mask=cached_graph_vec_mask)

            if i > 0:
                logits = self.out(enc_output[:, :-1, :])  # (b, l, n_vocab)
                enc_logits.append(logits.transpose(1, 2))

            # cache for later use.
            cached_post, cached_post_mask, cached_graph_vec, cached_graph_vec_mask = post, post_mask, graph_vec, graph_vec_mask

        ### Decode: supports unrolling.
        bsz, max_decode_len = response.size()
        dec_logits = torch.zeros(max_decode_len - 1, bsz, self.n_vocab).to(response.device)

        # SOS token
        dec_input = response[:, :1] # (b, 1)

        for t in range(max_decode_len - 1):
            dec_input = self.word_embedding(dec_input)

            # reuse encoder lstm hidden
            lstm_out, lstm_hidden = self.lstm(dec_input, lstm_hidden)

            step_output, *_ = self.attn(query=lstm_out, state_keyval=cached_post, knowledge_keyval=cached_graph_vec, state_mask=cached_post_mask, knowledge_mask=cached_graph_vec_mask)

            logit = self.out(step_output) # (b, 1, n_vocab)
            dec_logits[t] = logit.transpose(0,1)
            top1 = logit.max(-1)[1] # (b, 1)

            # stochastic teacher forcing
            if random.random() < teacher_force_ratio:
                dec_input = response[:, t + 1].unsqueeze(1) # ground truth
            else:
                dec_input = top1

        # TODO: inference decode logic.
            
        return enc_logits, dec_logits.permute(1, 2, 0)


class MultiSourceAttention(nn.Module):
    """ Luong general (bilinear) attention. """
    def __init__(self, dim):
        super().__init__()
        self.W_s = nn.Linear(dim, dim, bias=False) # state attention
        self.W_k = nn.Linear(dim, dim, bias=False) # knowledge attention (on graph vectors)
        self.W_msa = nn.Linear(dim*2, dim)
        self.W_mix = nn.Linear(dim*2, dim)
        
    def forward(self, query, state_keyval, knowledge_keyval, state_mask=None, knowledge_mask=None):
        # query: (bsz, len_q, d) // hidden of this sent
        # keyval: (bsz, len_kv, d) // hidden or graphvec of previous sent
        # mask: (bsz, 1, len_kv)

        state_attn = torch.bmm(self.W_s(query), state_keyval.transpose(1, 2)) # (bsz, len_q, len_kv)
        if state_mask is not None:
            state_attn.data.masked_fill_(state_mask, -float('inf'))
        state_attn = F.softmax(state_attn, dim=-1)
        state_context = torch.bmm(state_attn, state_keyval) # (bsz, len_q, d)

        knowledge_attn = torch.bmm(self.W_k(query), knowledge_keyval.transpose(1, 2)) # (bsz, len_q, len_kv)
        if knowledge_mask is not None:
            # knowledge_attn.data.masked_fill_(knowledge_mask, -float('inf'))
            knowledge_keyval.data.masked_fill_(knowledge_mask.squeeze(1).unsqueeze(-1), 0)
        knowledge_attn = F.softmax(knowledge_attn, dim=-1)
        knowledge_context = torch.bmm(knowledge_attn, knowledge_keyval) # (bsz, len_q, d)

        msa_context = self.W_msa(torch.cat([state_context, knowledge_context], dim=-1))
        # TODO: knowledge_context가 nan이더라도 msa_context는 nan이 아닐 수 있도록.

        # build attentional query
        combined = torch.cat([query, msa_context], dim=-1) # (bsz, len_q, 2d)
        output = torch.tanh(self.W_mix(combined)) # (bsz, len_q, d)

        return output, msa_context, state_attn, knowledge_attn


class GraphAttention(nn.Module):
    def __init__(self, d_embed, d_proj):
        super().__init__()
        self.W = nn.Linear(d_embed*3, d_embed*3, bias=False)
        self.proj = nn.Linear(d_embed*2, d_proj, bias=False)
    
    def forward(self, entity, mask=None):
        # entity: (bsz, len, n_triple, 3, d_embed)

        bsz, len, n_triple, _, dim = entity.size()
        entity = entity.view(-1, len, n_triple, 3*dim)

        h, _, t = entity.chunk(3, dim=-1) # [(b, l, n, d) ..]
        ht = torch.cat([h, t], dim=-1) # (b, l, n, 2d)
        ht = self.proj(ht) # (b, l, n, d)
        if mask is not None:
            ht.data.masked_fill_(mask.unsqueeze(-1), 0)

        h, r, t = self.W(entity).chunk(3, dim=-1) # [(b, l, n, d) ..]

        attn = (r * torch.tanh(h + t)).sum(-1) # (b, l, n)
        # if mask is not None:
        #     attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn, dim=-1) # (b, l, n)

        # # mask for knowledge attention
        # nan_mask = torch.isnan(attn).sum(-1) > 0 # (b, l)

        context = torch.sum(attn.unsqueeze(-1) * ht, dim=2) # (b, l, d_proj)
        return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--glove_path', type=str, default='data/glove.6B.200d.txt')
    parser.add_argument('--d_embed', type=int, default=200)
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--d_context', type=int, default=256) # msa context vector
    parser.add_argument('--n_word_vocab', type=int, default=100)
    parser.add_argument('--n_rel_vocab', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--max_decode_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.n_word_vocab += 4

    word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    with open('data/glove.6B.200d.txt', 'r') as inf:
        for line in inf:
            w = line.split()[0]
            if w not in word2idx:
                idx = len(word2idx)
                word2idx[w] = idx
        
            if len(word2idx) == args.n_word_vocab:
                break

    idx2word = {v:k for k, v in word2idx.items()}
    idx2word = OrderedDict(sorted(idx2word.items(), key=lambda t: t[0]))
    
    model = IEMSAModel(args, idx2word)
    model = model.to('cuda:0')

    bsz = args.batch_size
    max_t = 10
    max_n_triple = 3

    batch = {}
    for i in range(4):
        
        post_k = 'post_{}'.format(i+1)
        post_len_k = 'post_length_{}'.format(i+1)
        ent_k = 'entity_{}'.format(i+1)
        ent_len_k = 'entity_length_{}'.format(i+1)
        ent_mask_k = 'entity_mask_{}'.format(i+1)

        post_lengths = torch.randint(low=1, high=max_t, size=(bsz,)) # SW: variable name lengths -> post_lengths
        batch[post_len_k] = torch.LongTensor(post_lengths)
        max_post_t = post_lengths.max().item()

        batch[post_k] = torch.randint(high=args.n_word_vocab, size=(bsz, max_post_t))
        for b in range(bsz):
            batch[post_k][b, post_lengths[b].item():] = 0

        batch[ent_k] = torch.zeros((bsz, max_post_t, max_n_triple, 3), dtype=torch.long)
        batch[ent_mask_k] = torch.zeros((bsz, max_post_t, max_n_triple), dtype=torch.bool)
        batch[ent_len_k] = torch.empty(bsz, max_post_t)
        
        for t in range(post_lengths[b].item()):
            # 모든 triple 다 채움
            batch[ent_k][0, t, :, 0] = batch[post_k][0, t] # head
            batch[ent_k][0, t, :, 1] = 1 # rel
            batch[ent_k][0, t, :, 2] = torch.randint(high=args.n_word_vocab, size=(3,)) # tail
        batch[ent_len_k][0, :] = 3

        for t in range(post_lengths[b].item()):
            # 첫 두개 triple만 채움
            batch[ent_k][1, t, :2, 0] = batch[post_k][1, t] # head
            batch[ent_k][1, t, :2, 1] = 1 # rel
            batch[ent_k][1, t, :2, 2] = torch.randint(high=args.n_word_vocab, size=(2,)) # tail
        batch[ent_mask_k][1, :, -1] = 1
        batch[ent_len_k][1, :] = 2
        
        for t in range(post_lengths[-2].item()):
            # 하나만 채움
            batch[ent_k][-2, t, :1, 0] = batch[post_k][-2, t] # head
            batch[ent_k][-2, t, :1, 1] = 1 # rel
            batch[ent_k][-2, t, :1, 2] = torch.randint(high=args.n_word_vocab, size=(1,)) # tail
        batch[ent_mask_k][-2, :, -2:] = 1
        batch[ent_len_k][-2, :] = 1
        
        # last batch: no triple
        batch[ent_mask_k][-1, :, :] = 1
        batch[ent_len_k][-1, :] = 0

    batch['response'] = torch.randint(high=args.n_word_vocab, size=(bsz, max_t))
    batch = {key: val.to("cuda:0") for key, val in batch.items()}

    model(batch)
