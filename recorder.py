import time
import torch
from dataset import UNK_IDX, SOS_IDX, EOS_IDX, PAD_IDX


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
        self.epoch_pp = 0
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()

    def batch_end(self, batch_idx, batch_size, loss, pp):
        self.batch_end_time = time.time()
        self.batch_loss = loss
        self.batch_pp = pp
        self.epoch_loss += self.batch_loss * batch_size
        self.epoch_pp += self.batch_pp * batch_size
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
            self.writer.add_scalar(f'{self.mode}-Batch loss', self.batch_pp, batch_record_idx)
            self.writer.add_scalar(f'{self.mode}-Batch perplexity', self.batch_loss, batch_record_idx)
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
        self.writer.add_scalar(f'{self.mode}-Epoch perplexity', self.epoch_pp / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar(f'{self.mode}-Epoch time', self.epoch_time, self.epoch_idx)

    def log_text(self, output, batch):
        if self.mode == 'Val':
            print('===== SAMPLE =====')
            print()
            n = min(batch['response'].size()[0], 8)
            output = output[:n]
            text_idx = list()
            for batch_key in ['post', 'response']:
                text_idx.append(batch[batch_key][:n].cpu())
            text_idx.append(torch.max(output.cpu().detach(), 1)[1])
            text_all = list()
            for n, lines in enumerate(zip(*text_idx)):
                texts = [f'{n + 1}']
                print(n+1)
                for name, line in zip(['post', 'response', 'response_output'], lines):
                    texts.append(name)
                    print(f'    {name}')
                    line_text = []
                    for idx in line:
                        idx = idx.item()
                        if idx == EOS_IDX:
                            break
                        if idx != SOS_IDX and idx != PAD_IDX:
                            line_text.append(self.idx2word.get(idx, 'UNK'))
                    sentence = ' '.join(line_text)
                    texts.append(sentence)
                    print(f'        {sentence}')
                text_all.append(' - '.join(texts))
            print('==================')
            self.writer.add_text('Samples', ' / '.join(text_all), self.epoch_idx)

