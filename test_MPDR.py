import torch
import torch.nn as nn
import argparse

from utils import logger
from utils.solver_forward import MPDTester, MPRTester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import create_Env_dataloader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    save_path = opt.root_dir + f'checkpoints/seed_{opt.seed}/'

    logger.info('=> PyTorch Version: {}'.format(torch.__version__), root=save_path)
    logger.info(opt, root=save_path)

    # Environment initialization
    device, pin_memory, num_workers = init_device(opt.seed, opt.cpu, opt.gpu, opt.cpu_affinity, save_path)

    # Create the data loader
    _, val_loader = create_Env_dataloader(path='/home/zhizhen/MPC/data/'+f'Ns_{opt.Ns_max}/', 
                                          batch_size=opt.batch_size, 
                                          num_workers=num_workers, 
                                          device=device,
                                          Ns_max=opt.Ns_max,
                                          noise_power=opt.noise)
    
    # Define model
    pretrained = save_path + f'MPD_Ns{opt.Ns_max}/lath.pth'
    model_MPD = init_model(pretrained, path_Num=opt.Ns_max, w=64, h=64, save_path=save_path)
    model_MPD.to(device)

    model_MPR = []
    for Ns in range(opt.Ns_min, opt.Ns_max + 1):
        pretrained = save_path + f'MPR_Ns{Ns}/lath.pth'
        model = init_model(pretrained, path_Num=Ns, w=64, h=64, save_path=save_path)
        model.to(device)
        model_MPR.append(model)

    # Define loss function
    criterion = nn.MSELoss().to(device)
    
    # MPD forward
    loss, nmse, pred_paras, nls = MPDTester(model_MPD, device, criterion, save_path=save_path)(val_loader)

    # LDA train
    lda = LinearDiscriminantAnalysis(n_components=opt.Ns_max - opt.Ns_min)
    pred_paras_flat = pred_paras.reshape(pred_paras.shape[0], -1)
    lda.fit(pred_paras_flat, nls)

    # LDA forward
    pred_nls = lda.predict(pred_paras_flat)

    # MPR forward
    results = MPRTester(model_MPR, pred_nls, device, criterion, Ns_range=[opt.Ns_min, opt.Ns_max], save_path=save_path)(val_loader)
    logger.info('Results: ' + ', '.join([str(item) for item in results]), root=save_path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='/home/zhizhen/MPC/Baseline_MPC/', help='root directory')
    # parser.add_argument('--pretrained', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for initialization')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training')
    parser.add_argument('--cpu-affinity', type=str, default=None, help='CPU affinity, like "0xffff"')
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--rmse-thres', type=float, default=1, help='rmse threshold')
    parser.add_argument('--noise', type=float, default=0, help='power of noise' )
    parser.add_argument('--Ns-min', type=int, default=5, help='minimum number of paths')
    parser.add_argument('--Ns-max', type=int, default=10, help='maximum number of paths')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)