import time
import torch


class Recorder:
    def __init__(self, args, writer, idx2word):
        self.timestamp = args.timestamp
        self.log_interval = args.log_interval
        self.writer = writer
        self.idx2word = idx2word
        print(f'Record {self.timestamp}')

    def epoch_start(self, epoch_idx, is_train, loader):
        self.epoch_idx = epoch_idx
        self.mode = 'Train' if is_train else 'Val'
        self.batch_num = len(loader)
        self.dataset_size = len(loader.dataset)
        self.epoch_loss = 0
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()

    def batch_end(self, batch_idx, batch_size, loss):
        self.batch_end_time = time.time()
        self.batch_loss = loss
        self.epoch_loss += self.batch_loss * batch_size
        self.batch_time = self.batch_end_time - self.batch_start_time
        self.batch_start_time = time.time()
        if self.mode == 'Train' and batch_idx % self.log_interval == 0:
            print('Train Batch: {} [{}/{}({:.0f}%)] Loss:{:.4f} / Time:{:.4f}'.format(
                self.epoch_idx,
                batch_idx * batch_size, self.dataset_size,
                100. * batch_idx / self.batch_num,
                self.batch_loss,
                self.batch_time))
            batch_record_idx = (self.epoch_idx - 1) * (self.batch_num//self.log_interval) + batch_idx // self.log_interval
            self.writer.add_scalar(f'{self.mode}-Batch loss', self.batch_loss, batch_record_idx)
            self.writer.add_scalar(f'{self.mode}-Batch time', self.batch_time, batch_record_idx)

    def epoch_end(self):
        self.epoch_end_time = time.time()
        self.epoch_time = self.epoch_end_time - self.epoch_start_time
        print('====> {}: {} Average loss: {:.4f} / Time: {:.4f}'.format(
            self.mode,
            self.epoch_idx,
            self.epoch_loss / self.dataset_size,
            self.epoch_time))
        self.writer.add_scalar(f'{self.mode}-Epoch loss', self.epoch_loss / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar(f'{self.mode}-Epoch time', self.epoch_time, self.epoch_idx)

    def log_text(self, output, batch):
        if self.mode == 'Val':
            n = min(batch['response'].size()[0], 8)
            text_idx = list()
            for batch_key in ['post_1', 'post_2', 'post_3', 'post_4', 'response']:
                text_idx.append(batch[batch_key][:n].cpu())
            for logit in output[0]+[output[1]]:
                text_idx.append(torch.max(logit.cpu().detach(), 1)[1])
            for n, lines in enumerate(zip(*text_idx)):
                texts = [f'{n+1}']
                for line in lines:
                    texts.append(' '.join([self.idx2word[idx.item()] for idx in line if idx in idx > 3]))
                self.writer.add_text('Samples', ' - '.join(texts), self.epoch_idx)

