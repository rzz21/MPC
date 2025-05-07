import torch
import torch.nn as nn
import argparse

from utils import logger
from utils.solver_MPDR import Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import create_Env_dataloader

def main(opt):
    save_path = opt.root_dir + f'checkpoints/seed_{opt.seed}/{opt.mode}_Ns{opt.Ns}_epochs{opt.epochs}/'

    logger.info('=> PyTorch Version: {}'.format(torch.__version__), root=save_path)
    logger.info(opt, root=save_path)

    # Environment initialization
    device, pin_memory, num_workers = init_device(opt.seed, opt.cpu, opt.gpu, opt.cpu_affinity, save_path)

    # Create the data loader
    if opt.mode == 'MPC':
        data_path = '/home/zhizhen/MPC/data/' + 'Ns_M/'
    else:
        data_path = '/home/zhizhen/MPC/data/' + f'Ns_{opt.Ns}/'
    train_loader, val_loader = create_Env_dataloader(path=data_path, 
                                                     batch_size=opt.batch_size, 
                                                     num_workers=num_workers, 
                                                     device=device,
                                                     Ns_max=opt.Ns,
                                                     noise_power=opt.noise)
    
    # Define model
    model = init_model(opt.pretrained, path_Num=opt.Ns, w=64, h=64, save_path=save_path)
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Define optimizer and scheduler
    lr_init = 1e-3 if opt.scheduler == 'const' else 2e-3
    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    if opt.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=opt.epochs * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=5e-5)
        
    # Define trainer
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=opt.resume,
                      save_path=save_path)
        
    trainer.loop(epochs=opt.epochs,
                 train_loader=train_loader,
                 valid_loader=val_loader,
                 test_loader=val_loader)
    
    # Final test
    loss, nmse = Tester(model, device, criterion, save_path=save_path)(val_loader)

    logger.info(f"\n=! Final test loss: {loss:.3e}"
                f"\n         test nmse: {nmse:.3e}\n", root=save_path)

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='/home/zhizhen/MPC/Baseline_MPC/', help='Root directory')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for initialization')
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use')
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training')
    parser.add_argument('--cpu-affinity', type=str, default=None, help='CPU affinity, like "0xffff"')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler')
    parser.add_argument('--epochs', type=int, default=200, help='Number of total epochs to run')
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    # parser.add_argument('--conf-thres', type=float, default=5e-1, help='confidence threshold')
    # parser.add_argument('--rmse-thres', type=float, default=1, help='rmse threshold')
    # parser.add_argument('--out-thres', type=float, default=0, help='output threshold')
    parser.add_argument('--noise', type=float, default=0, help='power of noise' )
    parser.add_argument('--mode', type=str, default='MPC', help='mode of training')
    parser.add_argument('--Ns', type=int, default=10, help='number of paths')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parser_opt()
    main(opt)