import os
import math
from ast import literal_eval
from collections import OrderedDict
import functools
import pickle
import hashlib

from pathos.helpers import mp
from pathos.multiprocessing import ProcessingPool as Pool
import zarr
import jsonlines
from tqdm import tqdm
import numpy as np
import torch

from utils import line_count, pad_1d, pad_2d, append_storage, resize_storage

import ipdb


DEFAULT_VOCAB = ['_PAD', '_UNK', '_SOS', '_EOS']
DEFAULT_ENT = ['_PAD', '_NAF']
PAD_IDX, UNK_IDX, NAF_IDX, SOS_IDX, EOS_IDX = 0, 1, 1, 2, 3
NAF_TRIPLE = [NAF_IDX, NAF_IDX, NAF_IDX]


def get_dataloader(args,
                   data_path='data',
                   data_name='train',
                   batch_size=128,
                   shuffle=True,
                   num_workers=4):

    dataset = CommonsenseDialDataset(args, data_path, data_name)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            collate_fn=collate_fn
                                            )
    return data_loader


class CommonsenseDialDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path='data', data_name='train'):
        assert data_name in ['train', 'test', 'valid'], "Data name should be among ['train', 'test', 'valid']."
        self.args = args
        self.data_path = data_path
        self.data_dump = f'{self.data_path}/{data_name}set.zarr'
        self.vocab_file = f'{self.data_path}/vocab.pkl'

        self.rel2idx = self.make_rel_vocab()

        if not os.path.isfile(self.vocab_file):
            self.init_vocab()
        else:
            with open(f'{self.data_path}/vocab.pkl', 'rb') as vf:
                self.word2idx = pickle.load(vf)
        
        if not os.path.exists(self.data_dump):
            self.init_data(data_name)

        # load
        self.data = zarr.open(self.data_dump, mode='r')
        self.idx2word = OrderedDict([(v, k) for k, v in self.word2idx.items()])


    def init_vocab(self):
        raw_dict = open(f'{self.data_path}/resource.txt', 'r').read()
        raw_dict = literal_eval(raw_dict)
        # keys: 'csk_entities', 'dict_csk_entities', 'dict_csk', 'vocab_dict' (no use), 'csk_triples', 'dict_csk_triples'
        
        # update with DEFAULT_VOCAB
        # idx of each word/entity: glove에서의 idx + 4
        self.word2idx = OrderedDict([*zip(DEFAULT_VOCAB, range(len(DEFAULT_VOCAB)))])
        with open(f'{self.data_path}/glove.840B.300d.txt', 'r') as glove_f:
            for i, line in enumerate(glove_f):
                if i >= 30000:
                    break
                k = line.split()[0]
                self.word2idx[k] = len(self.word2idx)
        
        # Store vocab
        print(f'Vocab size: {len(self.word2idx)}')
        with open(f'{self.data_path}/vocab.pkl', 'wb') as df:
            pickle.dump(self.word2idx, df)

            
    def init_data(self, data_name):
        print(f'Initializing {data_name} data...')
        raw_dict = open(f'{self.data_path}/resource.txt', 'r').read()
        raw_dict = literal_eval(raw_dict)

        ent2idx = {k:v+len(DEFAULT_ENT) for k, v in raw_dict['dict_csk_entities'].items()} # 0: _PAD, 1: _NAF; others are mapped to +2
        default_ent = {k:v for k, v in zip(DEFAULT_ENT, range(len(DEFAULT_ENT)))}
        ent2idx.update(default_ent)

        idx2triple = {v:k for k, v in raw_dict['dict_csk_triples'].items()}

        def transform_triple_to_hrt(triple_idx):
            """ Transforms triple-idx (as a whole) to h/r/t format """
            if triple_idx == -1: # for response_triple
                return NAF_TRIPLE
            triple = idx2triple[triple_idx]
            h, r, t = triple.split(', ')
            return [ent2idx[h], self.rel2idx[r], ent2idx[t]]

        def process_file(root, inp):
            start_i, filename = inp
            n_sample = line_count(filename)
            
            post = np.zeros((n_sample, self.args.max_sentence_len), dtype=np.int32)
            post_length = np.zeros((n_sample), dtype=np.int32) # valid length (without pad)
            response = np.zeros((n_sample, self.args.max_sentence_len), dtype=np.int32)
            response_length = np.zeros((n_sample), dtype=np.int32)
            post_triple = np.zeros((n_sample, self.args.max_sentence_len), dtype=np.int32)
            triple = np.zeros((n_sample, self.args.max_sentence_len, self.args.max_triple_len, 3), dtype=np.int32)
            entity = np.zeros((n_sample, self.args.max_sentence_len, self.args.max_triple_len), dtype=np.int32)
            response_triple = np.zeros((n_sample, self.args.max_sentence_len, 3), dtype=np.int32)

            max_post_len, max_response_len, max_triple_len = 0, 0, 0

            with jsonlines.open(filename) as df:
                for i, line in enumerate(df):

                    pl, rl = len(line['post']), len(line['response'])
                    post_length[i] = pl
                    response_length[i] = rl

                    max_post_len = max(pl, max_post_len) 
                    max_response_len = max(rl, max_response_len)
                    max_triple_len = max([len(l) for l in line['all_triples']] + [max_triple_len])

                    post[i, :pl] = [self.get_word_idx(p) for p in line['post']]
                    response[i, :rl] = [self.get_word_idx(r) for r in line['response']]

                    post_triple[i, :pl] = np.array(line['post_triples']) # [0, 0, 1, 0, 2...]
                    response_triple[i, :rl] = [transform_triple_to_hrt(rt) for rt in line['response_triples']]
                    
                    # put NAF_TRIPLE/entity at index 0
                    triple[i] = pad_2d([[NAF_TRIPLE]] + [[transform_triple_to_hrt(t) for t in triples] for triples in line['all_triples']], length=(self.args.max_sentence_len, self.args.max_triple_len, 3))
                    entity[i] = pad_2d([[NAF_IDX]] + [[e + len(DEFAULT_ENT) for e in entities] for entities in line['all_entities']], length=(self.args.max_sentence_len, self.args.max_triple_len))
                    # entity[i] = np.where(tmp_entity > 0, tmp_entity + len(DEFAULT_ENT), 0)

                # dump to zarr
                root['post'][start_i : start_i+n_sample] = post
                root['post_length'][start_i : start_i+n_sample] = post_length
                root['response'][start_i : start_i+n_sample] = post
                root['response_length'][start_i : start_i+n_sample] = response_length
                root['post_triple'][start_i : start_i+n_sample] = post_triple
                root['triple'][start_i : start_i+n_sample] = triple
                root['entity'][start_i : start_i+n_sample] = entity
                root['response_triple'][start_i : start_i+n_sample] = response_triple
                
            return max_post_len, max_response_len, max_triple_len

        
        toread = [f'{self.data_path}/{data_name}set_pieces/{piece}' for piece in os.listdir(f'{self.data_path}/{data_name}set_pieces')]
        n_lines = sum([line_count(piece) for piece in toread])
        init_n_lines = math.ceil(n_lines / 10000) * 10000 # 마지막 조각 사이즈가 지정된 청크 사이즈보다 작아져서 나는 에러 방지

        root = zarr.open(f'{self.data_path}/{data_name}set.zarr', mode='w')
        post = root.zeros('post', shape=(init_n_lines, self.args.max_sentence_len), chunks=(10000, None), dtype='i4')
        post_length = root.zeros('post_length', shape=(init_n_lines,), chunks=(10000,), dtype='i4') # valid length (without pad)
        response = root.zeros('response', shape=(init_n_lines, self.args.max_sentence_len), chunks=(10000, None), dtype='i4')
        response_length = root.zeros('response_length', shape=(init_n_lines,), chunks=(10000,), dtype='i4')
        post_triple = root.zeros('post_triple', shape=(init_n_lines, self.args.max_sentence_len), chunks=(10000, None), dtype='i4')
        triple = root.zeros('triple', shape=(init_n_lines, self.args.max_sentence_len, self.args.max_triple_len, 3), chunks=(10000, None, None, None), dtype='i4')
        entity = root.zeros('entity', shape=(init_n_lines, self.args.max_sentence_len, self.args.max_triple_len), chunks=(10000, None, None), dtype='i4')
        response_triple = root.zeros('response_triple', shape=(init_n_lines, self.args.max_sentence_len, 3), chunks=(10000, None, None), dtype='i4')

        pool = Pool(min(len(toread), mp.cpu_count()))
        func = functools.partial(process_file, root)
        iterinp = [(i*self.args.init_chunk_size, filename) for i, filename in enumerate(toread)]
        max_post_lens, max_response_lens, max_triple_lens = zip(*tqdm(pool.imap(func, iterinp), total=len(iterinp)))
        
        max_post_len, max_response_len, max_triple_len = max(max_post_lens), max(max_response_lens), max(max_triple_lens)

        # trim remaining space
        post.resize(n_lines, max_post_len)
        post_length.resize(n_lines)
        response.resize(n_lines, max_response_len)
        response_length.resize(n_lines)
        post_triple.resize(n_lines, max_post_len)
        triple.resize(n_lines, max_post_len, max_triple_len, 3)
        entity.resize(n_lines, max_post_len, max_triple_len)
        response_triple.resize(n_lines, max_response_len, 3)

        print(f'Dumped {data_name} at: {self.data_path}/{data_name}set.zarr')
        

    def make_rel_vocab(self):
        # Don't dump; call every time
        rel2idx = {'_PAD': PAD_IDX, '_UNK': UNK_IDX, '_NAF': NAF_IDX} # 통일성 위해서 unk 둠
        with open(f'{self.data_path}/relation.txt', 'r') as relf:
            rel_dict = {line.strip(): i for i, line in enumerate(relf, start=len(rel2idx))}
            rel2idx.update(rel_dict)
        return rel2idx


    def __len__(self):
        return len(self.data['post'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.data.arrays()}

    def get_word_idx(self, word):
        return self.word2idx.get(word, UNK_IDX)
    

def collate_fn(batch):
    post = torch.tensor([s['post'] for s in batch]) # (bsz, pl)
    post_length = torch.tensor([s['post_length'] for s in batch]) # (bsz,)
    response = torch.tensor([s['response'] for s in batch]) # (bsz, rl)
    response_length = torch.tensor([s['response_length'] for s in batch]) # (bsz,)
    post_triple = torch.tensor([s['post_triple'] for s in batch]) # (bsz, pl)
    triple = torch.tensor([s['triple'] for s in batch]) # (bsz, pl, tl, 3) # NOTE: 원래는 pl보다 작지만 (valid-pl-with-triple이므로) 그냥 똑같이 pl로 둠
    entity = torch.tensor([s['entity'] for s in batch]) # (bsz, pl, tl)
    response_triple = torch.tensor([s['response_triple'] for s in batch]) # (bsz, rl, 3)

    # Sort in descending length order
    perm_idx = torch.sort(post_length, descending=True)
    post, post_length, response, response_length, post_triple, triple, entity, response_triple = \
        post[perm_idx], post_length[perm_idx], response[perm_idx], response_length[perm_idx], post_triple[perm_idx], triple[perm_idx], entity[perm_idx], response_triple[perm_idx]

    max_pl = post_length[-1]
    max_rl = torch.max(response_length)
    max_tl = torch.max((entity == 0).sum(-1))

    post = post[:, :max_pl]
    response = response[:, :max_rl]
    post_triple = post_triple[:, :max_pl]
    triple = triple[:, :max_pl, :max_tl]
    entity = entity[:, :max_pl, :max_tl]
    response_triple = response_triple[:, :max_rl]

    batched_data = {
        'post': post,
        'post_length': post_length,
        'response': response,
        'response_length': response_length,
        'post_triple': post_triple,
        'triple': triple,
        'entity': entity,
        'response_triple': response_triple,
    }

    # def padding(sent, length):
    #     """ Add sos and eos tokens, then pad sentence to length"""
    #     return ['_SOS'] + sent + ['_EOS'] + (['_PAD'] * (length - len(sent) - 2))

    return batched_data


if __name__ == "__main__":
    args = {'max_sentence_len': 150, 'max_triple_len': 50, 'init_chunk_size': 10000}
    class Args(object):
        def __init__(self, adict):
            self.__dict__.update(adict)
    args = Args(args)
    dataloader = get_dataloader(args=args, batch_size=2, shuffle=False)
    batch = iter(dataloader).next()

    for k, v in batch.items():
        print(k, v.shape)
