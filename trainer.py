import argparse
import dataloader
from model import IEMSAModel
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tensorboardX import SummaryWriter
from recorder import Recorder


def epoch(epoch_idx, is_train):
    model.train() if is_train else model.eval()
    loader = train_loader if is_train else val_loader
    recorder.epoch_start(epoch_idx, is_train, loader)
    for batch_idx, batch in enumerate(loader):
        batch_size = batch['response'].size()[0]
        batch = {key: val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output[1], batch['response'][:, 1:])
        for logit, batch_key in zip(output[0], ['post_2', 'post_3', 'post_4']):
            loss += criterion(logit, batch[batch_key][:, 1:])
        if is_train:
            loss.backward()
            optimizer.step()
        recorder.batch_end(batch_idx, batch_size, loss.item())
    recorder.log_text(output, batch)
    recorder.epoch_end()



def train():
    for epoch_idx in range(1, args.epochs + 1):
        epoch(epoch_idx, is_train=True)
        epoch(epoch_idx, is_train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--project', type=str, default='iemsa')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"))
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--glove_path', type=str, default='data/glove.6B.200d.txt')
    parser.add_argument('--d_embed', type=int, default=200)
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--d_context', type=int, default=256) # msa context vector
    parser.add_argument('--n_word_vocab', type=int, default=10000)
    parser.add_argument('--n_rel_vocab', type=int, default=45)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--max_decode_len', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader = dataloader.get_dataloader(args.data_dir, 'train', args.batch_size, args.num_workers)
    val_loader = dataloader.get_dataloader(args.data_dir, 'val', args.batch_size, args.num_workers)
    model = IEMSAModel(args, train_loader.dataset.idx2word)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    writer = SummaryWriter(f'{args.log_dir}/{args.project}_{args.timestamp}')
    recorder = Recorder(args, writer, train_loader.dataset.idx2word)

    train()
