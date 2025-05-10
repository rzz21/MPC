import torch
import torch.nn as nn
import argparse
import time
import math
import os

from utils import logger
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import create_Env_dataloader
from utils.statics import NMSE_evaluator
from utils.metrics import AverageMeter, para2pos, cal_metric_pos

def main(opt):
    save_path = opt.root_dir + f'checkpoints/small_net_Gain/seed_{opt.seed}/test_MPD/MPD_Ns{opt.Ns}_epochs{opt.epochs}_batch_size{opt.batch_size}_lr{opt.lr}'

    logger.info('=> PyTorch Version: {}'.format(torch.__version__), root=save_path)
    logger.info(opt, root=save_path)

    # Environment initialization
    device, pin_memory, num_workers = init_device(opt.seed, opt.cpu, opt.gpu, opt.cpu_affinity, save_path)

    # Create the data loader
    train_loader, val_loader = create_Env_dataloader(path='/home/zhizhen/MPC/data/'+f'Ns_{opt.Ns}/', 
                                          batch_size=opt.batch_size, 
                                          num_workers=num_workers, 
                                          device=device,
                                          Ns_max=opt.Ns,
                                          noise_power=opt.noise)
    
    # Define model
    model_MPR = init_model(opt.pretrained, path_Num=opt.Ns, w=64, h=64, save_path=save_path)
    model_MPR.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)
    
    # MPR forward
    results = MPRTester(model_MPR,  device, criterion, opt.Ns, save_path=save_path)(train_loader)
    logger.info('Results: ' + ', '.join([str(item) for item in results]), root=save_path)

class MPRTester:
    def __init__(self, model, device, criterion, Ns, print_freq=20, save_path=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.Ns = Ns
        self.print_freq = print_freq
        self.save_path = save_path

    def __call__(self, test_loader, rmse_thres=1, verbose=True):
        self.model.eval()
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

        for batch_idx, (data, label_para, pos, nl) in enumerate(data_loader):
            data, label_para = data.to(self.device), label_para.to(self.device)
            _, _, height, width = data.shape
            pred_para = self.model(data)
            
            # calculate pos
            for sample_idx in range(data.shape[0]):
                nl_sample = nl[sample_idx].cpu().detach().numpy().astype(int).item()
                data_sample = data[sample_idx].unsqueeze(0)
                pos_sample = pos[sample_idx].to(self.device)
                label_para_sample = label_para[sample_idx].unsqueeze(0)
                label_para_sample = label_para_sample[:, :nl_sample, 1:]
                pred_para_sample = pred_para[sample_idx, :, 1:]

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
                if error and chamdis > 300 and plot_error:
                    color = ['#FF0000', '#FFFF00', '#D2691E', '#008000', '#FFA500', '#4B0082', '#800080', '#00FFFF', '#000000', '#7FFF00', '#D19275', '#DAA520', '#FF69B4']
                    import matplotlib.pyplot as plt
                    plt.figure()
                    image = data_sample[0, ...].cpu().numpy()
                    plt.imshow(image[-1, ...])
                    pxy = pred_para_sample[0, :,:2].cpu().numpy()
                    label_para_sample = label_para_sample * torch.tensor((height, width), device=self.device) 
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='/home/zhizhen/MPC/Baseline_MPC/', help='root directory')
    parser.add_argument('--pretrained', type=str, default='/home/zhizhen/MPC/Baseline_MPC/checkpoints/small_net_Gain/seed_2025/MPD_Ns5_epochs100_batch_size8_lr0.002/last.pth', help='Path to pre-trained model')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for initialization')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training')
    parser.add_argument('--cpu-affinity', type=str, default=None, help='CPU affinity, like "0xffff"')
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--rmse-thres', type=float, default=1, help='rmse threshold')
    parser.add_argument('--noise', type=float, default=0, help='power of noise' )
    parser.add_argument('--Ns', type=int, default=5, help='number of paths')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)