import argparse
import datetime
import os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset import get_dataloader, PAD_IDX, NAF_IDX
from model import CCMModel, Baseline
from recorder import Recorder
from criterion import criterion, perplexity, baseline_criterion
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
import ipdb


def epoch(epoch_idx, is_train=True):
    model.train() if is_train else model.eval()
    loader = train_loader if is_train else val_loader
    if is_train and args.distributed:
        loader.sampler.set_epoch(epoch_idx)
    if recorder:
        recorder.epoch_start(epoch_idx, is_train, loader)
    for batch_idx, batch in enumerate(loader):
        batch_size = batch['response'].size()[0]
        batch = {key: val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()
        output, pointer_prob = model(batch)
        pointer_prob_target = (batch['response_triple'] != NAF_IDX).all(-1).to(torch.float)
        pointer_prob_target.data.masked_fill_(batch['response'] == 0, PAD_IDX)
        loss, nll_loss = criterion(output, batch['response'][:, 1:], pointer_prob, pointer_prob_target[:, 1:])
        pp = perplexity(nll_loss)
        if is_train:
            loss.backward()
            optimizer.step()
        if recorder:
            recorder.batch_end(batch_idx, batch_size, loss.item(), pp.item())
    if recorder:
        recorder.log_text(output, batch)
        recorder.epoch_end()
        return recorder.epoch_loss


def train():
    min_loss = float('inf')
    for epoch_idx in range(1, args.epochs + 1):
        epoch(epoch_idx, is_train=True)
        if args.local_rank == 0:
            loss = epoch(epoch_idx, is_train=False)
            if loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), 'best_model.pt')
                print(f'Saved the best model with loss {min_loss}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--project', type=str, default='ccm')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"))
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_access', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--teacher_forcing', type=float, default=1.0)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--t_embed', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_glove_vocab', type=int, default=30000)
    parser.add_argument('--n_entity_vocab', type=int, default=22590)
    parser.add_argument('--gru_layer', type=int, default=2)
    parser.add_argument('--gru_hidden', type=int, default=512)
    parser.add_argument('--max_sentence_len', type=int, default=150)
    parser.add_argument('--max_triple_len', type=int, default=50)
    parser.add_argument('--max_response_len', type=int, default=150)
    parser.add_argument('--data_piece_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1

    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        dist.init_process_group(backend='nccl',
                                init_method='env://')

    # Data loading code
    train_loader = get_dataloader(args, data_path=args.data_dir, data_name='train', batch_size=args.batch_size, num_workers=args.num_workers, distributed=args.distributed)
    val_loader = get_dataloader(args, data_path=args.data_dir, data_name='valid', batch_size=args.batch_size, num_workers=args.num_workers)
    # create model
    if not args.baseline:
        model = CCMModel(args, train_loader.dataset).to(device)
    else:
        model = Baseline(args).to(device)
        criterion = baseline_criterion
    optimizer = optim.Adam(model.parameters(), args.lr)
    if args.distributed:
        model = DDP(model)

    recorder = None
    if args.local_rank == 0:
        writer = SummaryWriter(f'{args.log_dir}/{args.project}_{"b" if args.baseline else "c"}_{args.timestamp}')
        recorder = Recorder(args, writer, train_loader.dataset.idx2word)

    train()
