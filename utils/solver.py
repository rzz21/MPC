import time
import os
import torch
from collections import namedtuple
import matplotlib.pylab as plt

from utils import logger
from utils.statics import AverageMeter

__all__ = ['Trainer', 'Tester']

field = ('nmse', 'epoch') # To be determined
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r'''
    train
    '''

    def __init__(self, model, device, optimizer, criterion, scheduler, mode, resume=None, save_path=None, print_freq=20, val_freq=10, test_freq=10):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.mode = mode
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
        self.dis = []
        self.fscore = []
        self.precision = []
        self.rmse = []
        self.nmse = []
        # NOTE : 注意修改result
        self.best_dis = Result()
        self.best_fscore = Result()
        self.best_precision = Result()
        self.best_rmse = Result()
        self.best_nmse = Result()

        self.tester = Tester(model, device, criterion, mode, print_freq=print_freq, save_path=save_path)
        self.test_loader = None

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
                self.test_loss, ssim = self.test(test_loader) # To be determined
                self.ssims.append((ep, ssim)) # To be determined
                self.test_losses.append((ep, self.test_loss.item()))
            else:
                ssim=None

            self._loop_postprocessing(ssim)

        self.plot_losses([[epoch for (epoch, loss) in self.train_losses], [epoch for (epoch, loss) in self.val_losses], [epoch for (epoch, loss) in self.test_losses]], # To be determined, 
                         [[loss for (epoch, loss) in self.train_losses], [loss for (epoch, loss) in self.val_losses], [loss for (epoch, loss) in self.test_losses]], # To be determined, 
                         'loss', 'epoch', 'loss', 'figs_loss.png', ['train_loss', 'val_loss','test_loss'])
        self.plot_losses([[epoch for (epoch, ssim) in self.ssims]],
                         [[ssim for (epoch, ssim) in self.ssims]],
                         'ssim', 'epoch', 'ssim', 'figs_ssim.png', ['ssim'])
        
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
        self.best_ssim = checkpoint['best_ssim'] # To be determined
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.ssims = checkpoint['ssims'] # To be determined
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
    
    def _loop_postprocessing(self, ssim):
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_ssim': self.best_ssim,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'ssims': self.ssims,
        }

        if ssim is not None:
            if self.best_ssim.ssim is None or ssim > self.best_ssim.ssim:
                self.best_ssim = Result(ssim, self.cur_epoch)
                state['best_ssim'] = self.best_ssim
                self._save(state, name='best_ssim.pth')

        self._save(state, name='last.pth')

        if self.best_ssim.ssim is not None:
            logger.info(f'=> Best ssim: {self.best_ssim.ssim:.3f} at epoch {self.best_ssim.epoch}\n', root=self.save_path)

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

    def __init__(self, model, device, criterion, mode, print_freq=20, save_path=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.mode = mode
        self.print_freq = print_freq
        self.save_path = save_path

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            if self.mode == 'MPR':
                loss, nmse, chamfdis, f1score, precision, rmse, detection_rate, \
                    dp, fp, pp, rp = \
                    self._iteration(test_loader)
                if verbose:
                    logger.info(f'=> Test chamfer distance: {chamfdis:.3f}, f1score: {f1score:.3f}, '
                                f'precision: {precision:.3f}, rmse: {rmse:.3f}, detection rate: {detection_rate:.3f}, '
                                f'dp: {dp:.3f}, fp: {fp:.3f}, pp: {pp:.3f}, rp: {rp:.3f}\n',
                                f'nmse of para: {nmse:.3f}, loss: {loss:.3f}\n',
                                root=self.save_path)
                return loss, nmse, chamfdis, f1score, precision, rmse, detection_rate, dp, fp, pp, rp    
            else:
                loss, nmse,  = self._iteration(test_loader)
                if verbose:
                    logger.info(f'=> Test nmse: {nmse:.3f}, loss: {loss:.3f}\n',
                                root=self.save_path)
                return loss, nmse
    
    def _iteration(self, data_loader):
        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()
        if self.mode == 'MPR':
            results_list = []
            catch_list = []
            iter_chamdis = AverageMeter('Iter chamfer dis')
            iter_f1score = AverageMeter('Iter f1score')
            iter_precision = AverageMeter('Iter precision')
            iter_rmse = AverageMeter('Iter rmse')

        for batch_idx, (data, label_para, pos, nl) in enumerate(data_loader):
            data, label = data.to(self.device), label.to(self.device)
            data = torch.clamp(data, min=40, max=140)
            label_clamp = torch.clamp(label, min=40, max=140)
            data = (data - 40) / 100
            label_clamp = (label_clamp - 40) / 100

            output = self.model(data)
            MSE_loss = self.criterion(output.squeeze(1), label_clamp)
            IG_loss = image_gradient_loss(output.squeeze(1), label_clamp)
            loss = MSE_loss + self.alpha*IG_loss

            output = output*100 + 40
            ssim = evaluator(output, label) #`ssim` to be determined

            iter_ssim.update(ssim)
            iter_MSE_loss.update(MSE_loss)
            iter_IG_loss.update(IG_loss)
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Test: [{batch_idx + 1}/{len(data_loader)}] '
                            f'ssim: {iter_ssim.avg:.3f} | '
                            f'MSE loss: {iter_MSE_loss.avg:.4f} |'
                            f'IG loss: {iter_IG_loss.avg:.4f} |'
                            f'Total loss: {iter_loss.avg:.4f} | '
                            f'time: {iter_time.avg:.3f}\n',
                            root=self.save_path)
                
        logger.info(f'=> Test ssim: {iter_ssim.avg:.3f}\n', root=self.save_path)

        return iter_loss.avg, iter_ssim.avg

        for i in range(num_pair):
            data, label = next(iter(data_loader))
            idx = torch.randint(0, data.size(0), (1,)).item()
            data, label = data[idx].to(self.device), label[idx].to(self.device)
            output = self.model(data.unsqueeze(0)).squeeze(0).squeeze(0)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # 设置坐标范围
            extent = [0, 200, 5000, 0]  # [xmin, xmax, ymax, ymin]

            # 绘制 label x IG
            label_x_IG = label[:, :-1] - label[:, 1:]
            im1 = axes[0, 0].imshow(label_x_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[0, 0].set_aspect(0.02)
            axes[0, 0].set_title('Label Image Gradient -- x')
            axes[0, 0].set_xlabel('Range (km)')
            axes[0, 0].set_ylabel('Depth (m)')
            fig.colorbar(im1, ax=axes[0, 0])  # 添加colorbar

            # 绘制 label y IG
            label_y_IG = label[:-1, :] - label[1:, :]
            im2 = axes[0, 1].imshow(label_y_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[0, 1].set_aspect(0.02)
            axes[0, 1].set_title('Label Image Gradient -- y')
            axes[0, 1].set_xlabel('Range (km)')
            axes[0, 1].set_ylabel('Depth (m)')
            fig.colorbar(im2, ax=axes[0, 1])  # 添加colorbar

            # 绘制 output x IG
            output_x_IG = output[:, :-1] - output[:, 1:]
            im3 = axes[1, 0].imshow(output_x_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[1, 0].set_aspect(0.02)
            axes[1, 0].set_title('Output Image Gradient -- x')
            axes[1, 0].set_xlabel('Range (km)')
            axes[1, 0].set_ylabel('Depth (m)')
            fig.colorbar(im3, ax=axes[1, 0])  # 添加colorbar

            # 绘制 output y IG
            output_y_IG = output[:-1, :] - output[1:, :]
            im4 = axes[1, 1].imshow(output_y_IG.detach().cpu().numpy(), cmap='jet_r', extent=extent, vmin=-0.02, vmax=0.02)
            axes[1, 1].set_aspect(0.02)
            axes[1, 1].set_title('Output Image Gradient -- y')
            axes[1, 1].set_xlabel('Range (km)')
            axes[1, 1].set_ylabel('Depth (m)')
            fig.colorbar(im4, ax=axes[1, 1])  # 添加colorbar

            plt.tight_layout()
            plt.show()

            fig.suptitle(f'Pair {i + 1}')
            fig.tight_layout()
            os.makedirs(os.path.join(self.save_path, 'IG_figs'), exist_ok=True)
            fig.savefig(os.path.join(self.save_path, 'IG_figs', f'pair_{i + 1}.png'))
            plt.close(fig)
            pass