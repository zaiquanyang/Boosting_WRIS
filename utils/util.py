from tabnanny import check
from tkinter.messagebox import NO
import torch
import torch.distributed as dist
from torch import Tensor
import os 
import numpy as np 
import psutil


def compute_mask_IU(masks, target):
    assert (target.shape[-2:] == masks.shape[-2:])
    # I = np.sum(np.logical_and(masks, target))
    # U = np.sum(np.logical_or(masks, target))
    I = torch.sum(torch.logical_and(masks, target), dim=-1).sum(dim=-1)
    U = torch.sum(torch.logical_or(masks, target), dim=-1).sum(dim=-1)
    return I, U

def compute_multi_thre_IU(masks, sam_masks):
    # assert (target.shape[-2:] == masks.shape[-2:])
    # I = np.sum(np.logical_and(masks, target))
    # U = np.sum(np.logical_or(masks, target))
    
    pred_masks = masks * sam_masks
    
    sam_pred_IoUs = torch.zeros(5, len(sam_masks)).cuda()

    for k, thre in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
        pred_masks_bool = (pred_masks>thre)
        I = torch.sum(torch.logical_and(pred_masks_bool, sam_masks), dim=-1).sum(dim=-1)
        U = torch.sum(torch.logical_or(pred_masks_bool, sam_masks), dim=-1).sum(dim=-1)
        sam_pred_IoU = I*1.0/(U + 0.001)
        sam_pred_IoUs[k]= sam_pred_IoU
    max_IoU_id = torch.argmax(sam_pred_IoUs[:].sum(dim=0))
    return max_IoU_id



def reduce_tensor(args, tensor, mode='mean'):
    rt = tensor.clone()
    # rt = tensor    ###### 
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= args.world_size      # word_size = the number of gpus 
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt


class AverageMeter:
    """
    Compute and stores the average and current value
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(epoch,model,optimizer,lr_schdeduler,logger=None,args=None, checkpoint_name=None):
    try:
        save_state={
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'lr_scheduler':lr_schdeduler.state_dict(),
            'epoch':epoch
        }
    except:
        save_state={
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'lr_scheduler':lr_schdeduler,
        'epoch':epoch
    }

    if not os.path.exists(args.output):
        print('mkdir ', args.output)
        os.makedirs(args.output)

    if checkpoint_name is None:
        checkpoint_name = f'ckpt_448_epoch_{epoch}.pth'

    save_path=os.path.join(args.output, checkpoint_name)
    torch.save(save_state, save_path)
    print(f"{save_path} saved !!!")

    return save_path 

# return start epoch
# load LAVT model
def load_checkpoint(args,model_without_ddp,optimizer=None,lr_scheduler=None,logger=None):
    root_path=args.output
    pretrain_name=args.pretrain
    print(os.path.join(root_path,pretrain_name), '-------')
    checkpoint=torch.load(os.path.join(root_path,pretrain_name),map_location='cpu')

    msg=model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    # print(msg)
    # resume not evaluation
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1 
    print(f"=> loaded successfully '{pretrain_name}'")
    del checkpoint
    torch.cuda.empty_cache()

def load_pretrained_checkpoint(name,model_without_ddp):
    # root_path=args.output
    # pretrain_name=args.pretrain
    # print(os.path.join(root_path,pretrain_name), '-------')
    checkpoint=torch.load(name, map_location='cpu')

    msg=model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    print(msg)
    del checkpoint
    torch.cuda.empty_cache()

# load pretrained swin transformer
def load_pretrained_swin(config,model,logger):
    logger.info(f"==============> Loading weight {config.PRETRAIN.PATH} for fine-tuning......")
    checkpoint=torch.load(config.PRETRAIN.PATH,map_location='cpu')
    state_dict=checkpoint['model']


    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")


    # for k in state_dict.keys():
    #     print(k)

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)




def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used