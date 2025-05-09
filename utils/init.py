import os
import random
import thop
import torch
import multiprocessing

from model import MPDR
from utils import logger, line_seg

__all__ =["init_device", "init_model"]

def init_device(seed=None, cpu=None, gpu=None, affinity=None, save_path=None):
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0), root=save_path)
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU", root=save_path)

    num_workers = multiprocessing.cpu_count()
    logger.info("num workers:%d" % num_workers, root=save_path)

    return device, pin_memory, num_workers

def init_model(pretrained, path_Num, w=64, h=64, save_path=None):
    model = MPDR(c=2, w=w, h=w, param_num=3, path_Num=path_Num)

    if pretrained is not None:
        assert os.path.isfile(pretrained)
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(pretrained), root=save_path)

    image = torch.randn([1, 2, h, w])
    flops, params = thop.profile(model, inputs=(image, ), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    logger.info(f'=> Model Name: MPDR [pretrained: {pretrained}]', root=save_path)
    logger.info(f'=> Model FLOPs: {flops}', root=save_path)
    logger.info(f'=> Model Params: {params}', root=save_path)
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n', root=save_path)

    return model

