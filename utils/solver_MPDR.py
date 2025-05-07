import time
import os
import torch
from collections import namedtuple
import matplotlib.pylab as plt

from utils import logger
from utils.statics import AverageMeter, NMSE_evaluator

__all__ = ['Trainer', 'Tester']

field = ('nmse', 'epoch') # To be determined
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r'''
    train
    '''

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None, save_path=None, print_freq=50, val_freq=10, test_freq=10):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.nmse = []
        self.best_nmse = Result()

        self.tester = Tester(model, device, criterion, print_freq=print_freq, save_path=save_path)

    def loop(self, epochs, train_loader, valid_loader, test_loader):
        r'''
        each epoch
        '''

        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # train
            self.train_loss = self.train(train_loader)
            self.train_losses.append((ep, self.train_loss.item()))

            # val
            if ep % self.val_freq == 0:
                self.val_loss = self.val(valid_loader)
                self.val_losses.append((ep, self.val_loss.item()))

            # test
            if ep % self.test_freq == 0:
                self.test_loss, nmse = self.test(test_loader)
                self.nmse.append((ep, nmse)) 
                self.test_losses.append((ep, self.test_loss.item()))
            else:
                nmse=None

            self._loop_postprocessing(nmse)

        self.plot_losses([[epoch for (epoch, loss) in self.train_losses], [epoch for (epoch, loss) in self.val_losses], [epoch for (epoch, loss) in self.test_losses]], # To be determined, 
                         [[loss for (epoch, loss) in self.train_losses], [loss for (epoch, loss) in self.val_losses], [loss for (epoch, loss) in self.test_losses]], # To be determined, 
                         'loss', 'epoch', 'loss', 'figs_loss.png', ['train_loss', 'val_loss','test_loss'])
        # self.plot_losses([[epoch for (epoch, nmse) in self.nmse]],
        #                  [[nmse for (epoch, nmse) in self.nmse]],
        #                  'nmse', 'epoch', 'nmse', 'figs_nmse.png', ['nmse'])
        
    def _resume(self):
        if self.resume_file is None:
            return None
        
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}', root=self.save_path)
        checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_nmse = checkpoint['best_nmse'] # To be determined
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.nmse = checkpoint['nmse'] # To be determined
        self.cur_epoch += 1

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} from epoch {self.cur_epoch}.\n', root=self.save_path)

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No save path specified, checkpoint not saved.')
            return
        
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def train(self, train_loader):
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)
        
    def val(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(valid_loader)
        
    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=True)
        
    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Ier time')
        time_tmp = time.time()

        for batch_idx, (data, label_para, _, nl) in enumerate(data_loader):
            data, label_para = data.to(self.device), label_para.to(self.device)
            pred_para = self.model(data)
            loss = self.criterion(pred_para, label_para)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                            f'MSE loss: {iter_loss.avg:.3e} | '
                            f'time: {iter_time.avg:.3f}\n',
                            root=self.save_path)
                
        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                    f' {mode} Loss: {iter_loss.avg:.3e} \n',
                    root=self.save_path)
        
        return iter_loss.avg
    
    def _loop_postprocessing(self, nmse):
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_nmse': self.best_nmse,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'nmse': self.nmse,
        }

        if nmse is not None:
            if self.best_nmse.nmse is None or nmse < self.best_nmse.nmse:
                self.best_nmse = Result(nmse, self.cur_epoch)
                state['best_nmse'] = self.best_nmse
                self._save(state, name='best_nmse.pth')

        self._save(state, name='last.pth')

        if self.best_nmse.nmse is not None:
            logger.info(f'=> Best nmse: {self.best_nmse.nmse:.3f} at epoch {self.best_nmse.epoch}\n', root=self.save_path)

    def plot_losses(self, x_values, y_values, title, xlabel, ylabel, filename, labels):
        fig, ax = plt.subplots(figsize=(20, 10))
        for x, y, label in zip(x_values, y_values, labels):
            ax.semilogy(x, y, linewidth=3, marker='o', markersize=10, label=label)
        ax.set_title(title, fontsize=40)
        ax.set_xlabel(xlabel, fontsize=30)
        ax.set_ylabel(ylabel, fontsize=30)
        ax.legend(fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)

        if self.save_path is None:
            logger.warning('No save path specified, plot not saved.')
            return
        
        os.makedirs(self.save_path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, filename))


class Tester:
    r'''
    test
    ssim  to be determined
    '''

    def __init__(self, model, device, criterion, print_freq=20, save_path=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        self.save_path = save_path

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, nmse  = self._iteration(test_loader)
            if verbose:
                logger.info(f'=> Test nmse: {nmse:.3f}, loss: {loss:.3f}\n',
                            root=self.save_path)
        return loss, nmse
    
    def _iteration(self, data_loader):
        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (data, label_para, _, nl) in enumerate(data_loader):
            data, label_para = data.to(self.device), label_para.to(self.device)
            pred_para = self.model(data)
            loss = self.criterion(pred_para, label_para)

            nmse = NMSE_evaluator(pred_para, label_para)
            iter_nmse.update(nmse)
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Test: [{batch_idx + 1}/{len(data_loader)}] '
                            f'NMSE: {iter_nmse.avg:.3f} | '
                            f'Total loss: {iter_loss.avg:.4f} | '
                            f'time: {iter_time.avg:.3f}\n',
                            root=self.save_path)
                
        logger.info(f'=> Test NMSE: {iter_nmse.avg:.3f} | '
                    f'loss: {iter_loss.avg:.3f} \n',
                    root=self.save_path)
        return iter_loss.avg, iter_nmse.avg