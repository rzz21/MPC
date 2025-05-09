import time
import os
import torch
from collections import namedtuple
import matplotlib.pylab as plt
import numpy
import math

from utils import logger
from utils.statics import AverageMeter, NMSE_evaluator
from utils.metrics import AverageMeter, para2pos, cal_metric_pos

__all__ = ['MPDTester', 'MPRTester']

class MPDTester:
    def __init__(self, model, device, criterion, print_freq=20, save_path=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        self.save_path = save_path

    def __call__(self, test_loader, verbose=True):
        self.model.eval()
        with torch.no_grad():
            loss, nmse, pred_paras, nls = self._iteration(test_loader)
            if verbose:
                logger.info(f'=> Test nmse: {nmse:.3f}, loss: {loss:.3f}\n',
                            root=self.save_path)
        return loss, nmse, pred_paras, nls

    def _iteration(self, data_loader):
        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (data, label_para, _, nl) in enumerate(data_loader):
            data, label_para = data.to(self.device), label_para.to(self.device)
            pred_para = self.model(data)
            
            if batch_idx == 0:
                pred_paras = pred_para.cpu().detach().numpy()
                nls = nl.cpu().detach().numpy()
            else:
                pred_paras = numpy.concatenate((pred_paras, pred_para.cpu().detach().numpy()), axis=0)
                nls = numpy.concatenate((nls, nl.cpu().detach().numpy()), axis=0)
            
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
        return iter_loss.avg, iter_nmse.avg, pred_paras, nls
    
class MPRTester:
    def __init__(self, model_MPR, pred_nls, device, criterion, Ns_range, print_freq=20, save_path=None):
        self.model_MPR = model_MPR
        self.pred_nls = pred_nls
        self.device = device
        self.criterion = criterion
        self.Ns_range = Ns_range
        self.print_freq = print_freq
        self.save_path = save_path

    def __call__(self, test_loader, rmse_thres=1, verbose=True):
        for model in self.model_MPR:
            model.eval()
        with torch.no_grad():
            results = self._iteration(test_loader, rmse_thres)
            if verbose:
                logger.info('Test results: ' + ', '.join([str(item) for item in results]), root=self.save_path)
        return results

    def _iteration(self, data_loader, rmse_thres):
        results_list = []
        catch_list = []
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()
        iter_chamdis = AverageMeter("chamfer distance") 
        iter_f1score = AverageMeter("f1-score")
        iter_precision = AverageMeter("precision")
        iter_rmse = AverageMeter("rmse")

        # batch_size
        for batch_idx, (data, label_para, pos, nl) in enumerate(data_loader):
            _, nc, height, width = data.shape
            for sample_idx in range(data.shape[0]):
                total_idx = sample_idx + batch_idx * data.shape[0]
                nl_sample = nl[sample_idx].cpu().detach().numpy().astype(int).item()
                data_sample = data[sample_idx].unsqueeze(0).to(self.device)
                pos_sample = pos[sample_idx].to(self.device)
                label_para_sample = label_para[sample_idx].unsqueeze(0).to(self.device)
                label_para_sample = label_para_sample[:, :nl_sample, :]

                # decide which model to use
                Ns = self.pred_nls[total_idx]
                model_idx = Ns - self.Ns_range[0]
                pred_para_sample = self.model_MPR[model_idx](data_sample).squeeze(0)

                # process pred_para
                pred_para_sample[:, 0] = pred_para_sample[:, 0] + (1 / height)
                pred_para_sample = pred_para_sample * height

                # calculate pos
                pos_user_sample = pos_sample[0, :]
                label_pos_sample = pos_sample[1: nl_sample + 1, :]
                pred_pos_sample = para2pos(pred_para_sample, pos_user_sample)
                chamdis, f1score, precision, error, rmse = cal_metric_pos(label_pos_sample, pred_pos_sample, threshold=1, rmse_thres=rmse_thres)

                # debug 
                plot_error = False
                label_para_sample = label_para_sample * torch.tensor((height, width), device=self.device) 
                if error and chamdis > 300 and plot_error:
                    color = ['#FF0000', '#FFFF00', '#D2691E', '#008000', '#FFA500', '#4B0082', '#800080', '#00FFFF', '#000000', '#7FFF00', '#D19275', '#DAA520', '#FF69B4']
                    import matplotlib.pyplot as plt
                    plt.figure()
                    image = data_sample[0, ...].cpu().numpy()
                    plt.imshow(image[-1, ...])
                    pxy = pred_para_sample[0, :,:2].cpu().numpy()
                    lxy = label_para_sample[0, ...].detach().cpu().numpy()
                    px, py = pxy.T
                    lx, ly = lxy.T
                    plt.scatter(lx, ly, c='b', marker='o', label='label')
                    for j in range(len(px)):
                        plt.scatter(px[j], py[j], marker='x', color=color[j])
                    plt.show()
                    plt.savefig(os.path.join(self.save_path, f'error_plot/batch{batch_idx}_si{sample_idx}_H.jpg'))
                    plt.close()
                    
                    plt.figure()
                    pxx, pyy = pred_pos_sample.detach().cpu().numpy().T
                    lxx, lyy = label_pos_sample.detach().cpu().numpy().T
                    plt.scatter(lxx, lyy)
                    for j in range(len(pxx)):
                        plt.scatter(pxx[j], pyy[j], marker='x', color=color[j])              
                    p_u = pos_user_sample.detach().cpu().numpy()
                    plt.scatter(p_u[0], p_u[1], marker='s', color='green')
                    plt.ylim([-50,50])
                    plt.xlim([0,100])
                    plt.show()
                    plt.savefig(os.path.join(self.save_path, f'error_plot/batch{batch_idx}_si{sample_idx}_H.jpg'))
                    plt.close()
                    # print('debug')
                
                results_list.append(torch.stack((chamdis, f1score, precision, torch.sqrt(rmse.nanmean()))))
            
                catch_list.append(not error)
                iter_chamdis._update(chamdis)
                iter_f1score._update(f1score)
                iter_precision._update(precision)
                iter_rmse._update(rmse)
                rmse_avg = torch.sqrt(iter_rmse.avg) if isinstance(iter_rmse.avg, torch.Tensor) else math.sqrt(iter_rmse.avg)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            
        logger.info(f"For all cases:", root=self.save_path)
        logger.info(('%15s' * 4) % ('Chamfer Dis', 'F1-score', 'Precision', 'RMSE'), root=self.save_path)
        rmse_avg = torch.sqrt(iter_rmse.avg) if isinstance(iter_rmse.avg, torch.Tensor) else math.sqrt(iter_rmse.avg)
        logger.info(('%15.4f' * 4) % tuple([iter_chamdis.avg, iter_f1score.avg, iter_precision.avg, rmse_avg]), root=self.save_path)            

        # print results
        catch_result_list = [results_list[i] for i in range(len(catch_list)) if catch_list[i]]
        detection_rate, chamdis_p, fscore_p, precision_p, rmse_p = 0, 10, 0, 0, 10
        
        if len(catch_result_list) > 0:
            catch_result = torch.stack(catch_result_list)
            detection_rate = sum(catch_list) / len(catch_list)
            logger.info(f'Detection rate: {detection_rate:.2f}', root=self.save_path)
            logger.info(f'For detected cases:', root=self.save_path)  
            logger.info(('%15s' * 4) % ('Chamfer Dis', 'F1-score', 'Precision', 'RMSE'), root=self.save_path)
            logger.info(('%15.4f' * 4) % tuple(catch_result.nanmean(dim=0).tolist()), root=self.save_path)
            chamdis_p, fscore_p, precision_p, rmse_p = catch_result.nanmean(dim=0).tolist()
        else:   
            logger.info('No detected cases', root=self.save_path)

        rmse_avg = torch.sqrt(iter_rmse.avg) if isinstance(iter_rmse.avg, torch.Tensor) else math.sqrt(iter_rmse.avg)
        return iter_chamdis.avg, iter_f1score.avg, iter_precision.avg, rmse_avg, detection_rate, chamdis_p, fscore_p, precision_p, rmse_p
                