from cProfile import label
import os
from sched import scheduler
import sched
from typing import IO
# os.environ['CUDA_ENABLE_DEVICES'] = '2,3'

import torch 
import os
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from dataset.transform import get_transform
from args import get_parser
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.util import AverageMeter, load_checkpoint, save_checkpoint, load_pretrained_checkpoint,get_gpu_mem_info, get_cpu_mem_info
import time 
from logger import create_logger
import datetime
import numpy  as np 
import torch.nn as nn 
from tensorboardX import SummaryWriter
import CLIP.clip as clip 
import torchvision 
from torch.cuda.amp import autocast as autocast, GradScaler

# from model.model_stage1_v0 import TRIS 
# from model.model_stage1_v2 import TRIS
# from model.model_prog_v1 import TRIS
# from model.model_prog_v2 import TRIS
# from model.model_prog_v4 import TRIS
from model.model_prog_v5 import TRIS
from dataset.ReferDataset import ReferDataset 
from validate import validate 


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seed(1234)


def main(args):
    if args.distributed:
        local_rank=dist.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    model = TRIS(args)
    try:
        param_groups = model.trainable_parameters()
    except:
        print() 
        param_groups = None 
        print('no param goups...')
        print() 
    if args.distributed:
        model.cuda(local_rank)
    else:
        model.cuda() 
    # #################
    # for param in model.backbone.parameters():
    #     param.require_grad = False 
    # #################

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    else:
        model=torch.nn.DataParallel(model)

    model_without_ddp=model.module
    print() 
    # print(model_without_ddp)
    print()
    # model.train() 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params / 1e6: .2f}M")
    # build dataset
    train_dataset = ReferDataset(refer_data_root=args.refer_data_root,
                                dataset=args.dataset,
                                splitBy=args.splitBy,
                                bert_tokenizer=args.bert_tokenizer,
                                split='train',
                                size=args.size,
                                max_tokens=args.max_query_len,
                                image_transforms=get_transform(args.size, 
                                                    train=True),
                                eval_mode=args.eval,
                                negative_samples=args.negative_samples,
                                positive_samples=args.positive_samples)
    val_datasets = [] 
    for test_split in args.test_split.split(','):
        val_datasets.append(ReferDataset(refer_data_root=args.refer_data_root,
                                            dataset=args.dataset,
                                            splitBy=args.splitBy,
                                            bert_tokenizer=args.bert_tokenizer,
                                            split=test_split,
                                            size=args.size,
                                            max_tokens=args.max_query_len,
                                            image_transforms=get_transform(args.size, train=False),
                                            eval_mode=True,
                                            scales=args.scales,
                                            positive_samples=args.positive_samples)) 
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(DistributedSampler(val_dataset, shuffle=False))
    else:
        train_sampler = None 
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(None)

    train_loader=DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            sampler=train_sampler,
                            shuffle=(train_sampler is None))
    val_loaders = [] 
    for val_dataset, val_sampler in zip(val_datasets, val_samplers):
        val_loaders.append(DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=2,
                            pin_memory=True, 
                            sampler=val_sampler,
                            shuffle=False))

    if param_groups is not None:
        print('param_groups is Not None !')
        optimizer = AdamW([
            {'params': param_groups[0], 'lr': args.lr * args.lr_multi, 'weight_decay': args.weight_decay},
            {'params': param_groups[1], 'lr': args.lr, 'weight_decay': args.weight_decay},
        ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(params=model.parameters(), 
                      lr=args.lr, 
                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda x: (1 - x / (len(train_loader) * args.epoch)) ** 0.9)
    
    if args.resume:
        if args.pretrain is not None:
            load_checkpoint(args, model_without_ddp, optimizer, scheduler, logger)  #####
        if args.eval:
            st = time.time()
            val_acc, testA_acc, testB_acc = 0, 0, 0
            for i, val_loader in enumerate(val_loaders):
                oIoU, mIoU, hit = validate(args, val_loader, model, local_rank)
                if i == 0: val_acc = mIoU
                elif i == 1: testA_acc = mIoU
                else: testB_acc = mIoU
            print(f'val: {val_acc}, testA, {testA_acc}, testB: {testB_acc}')
            all_t = time.time() - st 
            print(f'Testing time:  {str(datetime.timedelta(seconds=int(all_t)))}')
            return
    
    logger.info("\nStart training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_max_len = args.max_query_len
    # clip_model, _ = clip.load("../clip_weights/ViT-B-32.pt", device=device, jit=False, txt_length=tmp_max_len)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, txt_length=tmp_max_len)
    clip_model.eval()

    train_time = 0
    start_time = time.time()
    best = {
        'val_acc': -1,
        'val_hit': -1,
        'epoch': -1,
        'path': '',
        'hit': -1,
        'hit_path': '',
        'testA': -1,
        'testB': -1 
    }
    iteration = 0
    scaler = GradScaler(enabled=args.amp_training)
    for epoch in range(args.start_epoch, args.epoch):
        st = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        iteration = train_one_epoch(train_loader, model, optimizer, epoch, local_rank, args, iteration, clip_model, lr_scheduler=scheduler, amp_scaler=scaler, config=args)

        train_time += time.time() - st 

        # validation, save best 
        val_mIoU, testA_mIoU, testB_mIoU = 0, 0, 0
        val_oIoU, testA_oIoU, testB_oIoU = 0, 0, 0
        val_hit, testA_hit, testB_hit = 0, 0, 0
        val_hitM, testA_hitM, testB_hitM = 0, 0, 0
        zs_val_mIoU, zs_tA_mIoU, zs_tB_mIoU = 0., 0., 0.
        for i, val_loader in enumerate(val_loaders):
            oIoU, mIoU, hit, zs_mIoU, zs_oIoU, Hit_M = validate(args, val_loader, model, local_rank)
            if i == 0: val_mIoU, val_oIoU, val_hit, val_hitM, zs_val_mIoU = mIoU, oIoU, hit, Hit_M, zs_mIoU
            elif i == 1: testA_mIoU, testA_oIoU, testA_hit, testA_hitM, zs_tA_mIoU = mIoU, oIoU, hit, Hit_M, zs_mIoU
            else: testB_mIoU, testB_oIoU, testB_hit, testB_hitM, zs_tB_mIoU = mIoU, oIoU, hit, Hit_M, zs_mIoU
            
        if val_mIoU > best['val_acc'] and local_rank==0:
            if os.path.exists(best['path']):
                print('remove ', best['path'])
                os.remove(best['path'])
            save_path = save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args, f'ckpt_320_epoch_{epoch}_best.pth')
            best['val_acc'] = val_mIoU.item() 
            best['val_hit'] = val_hit 
            best['epoch'] = epoch 
            best['path'] = save_path
            best['testA'] = testA_mIoU
            best['testB'] = testB_mIoU
            

        if val_hit > best['hit'] and local_rank==0:
            best['hit'] = hit 
            if os.path.exists(best['hit_path']):
                print('remove ', best['hit_path'])
                os.remove(best['hit_path'])
            save_path = save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args, f'ckpt_320_epoch_{epoch}_hit.pth')
            best['hit_path'] = save_path
            best['hit'] = val_hit 
        print(best)

        if local_rank == 0:
            writer.add_scalar('mIoU/val', val_mIoU, epoch)
            writer.add_scalar('mIoU/testA', testA_mIoU, epoch)
            writer.add_scalar('mIoU/testB', testB_mIoU, epoch)
            writer.add_scalar('mIoU/ZS_Val_mIoU', zs_val_mIoU, epoch)
            writer.add_scalar('mIoU/ZS_tA_mIoU', zs_tA_mIoU, epoch)
            writer.add_scalar('mIoU/ZS_tB_mIoU', zs_tB_mIoU, epoch)


            writer.add_scalar('oIoU/ZS_oIoU', zs_oIoU, epoch)
            writer.add_scalar('oIoU/val', val_oIoU, epoch)
            writer.add_scalar('oIoU/testA', testA_oIoU, epoch)
            writer.add_scalar('oIoU/testB', testB_oIoU, epoch)

            writer.add_scalar('Hit/val', val_hit, epoch)
            writer.add_scalar('Hit/testA', testA_hit, epoch)
            writer.add_scalar('Hit/testB', testB_hit, epoch)

            writer.add_scalar('HitM/val', val_hitM, epoch)
            writer.add_scalar('HitM/testA', testA_hitM, epoch)
            writer.add_scalar('HitM/testB', testB_hitM, epoch)
        
        if epoch % 4 == 0:
            save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args, f'ckpt_320_epoch_{epoch}.pth')
    print()
    print()
    # last_trainset = ReferDataset(refer_data_root=args.refer_data_root,
    #                         dataset=args.dataset,
    #                         split='train',
    #                         splitBy=args.splitBy,
    #                         image_transforms=get_transform(args.size, train=False),
    #                         eval_mode=True,
    #                         size=args.size,
    #                         bert_tokenizer=args.bert_tokenizer)
    # val_train_loader = DataLoader(last_trainset,
    #                         batch_size=1,
    #                         num_workers=2,
    #                         pin_memory=True, 
    #                         sampler=val_sampler)
    # print('loading ', best['path'])
    # load_pretrained_checkpoint(best['path'], model_without_ddp)
    # oIoU_1, mIoU_1, hit_1 = validate(args, val_train_loader, model, local_rank)
    # print('Validat on the train split: ', oIoU_1, mIoU_1, hit_1)
    # print(best)
    # # #########
    # print()
    # print()
    # print('--------same sents--------')
    # print()
    # from validate import validate_same_sentence 
    # oIoU, mIoU, hit = validate_same_sentence(args, val_train_loader, model, local_rank, save_cam=False)
    # print() 
    # print('Validat on the train split (same sents): ', oIoU, mIoU, hit)
    # print('Validat on the train split: ', oIoU_1, mIoU_1, hit_1)
    # print(best)
    # # ##################

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(train_time))
    logger.info('Training + testing time {}'.format(total_time_str))


def train_one_epoch(train_loader,model,optimizer,epoch,local_rank,args, iteration=0, clip_model=None, lr_scheduler=None, amp_scaler=None, config=None):
    num_steps=len(train_loader)
    model.train()

    batch_time=AverageMeter()
    loss_meter=AverageMeter()

    start=time.time()
    end=time.time()

    max_iter = int(num_steps * args.epoch) 
    # print('='*20, ',  max_iter = ', max_iter)
    clip_input_size = 224 
    l1, l2, l3, l4, l5 = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
    
    for idx,(samples, targets) in enumerate(train_loader):
        img = samples['img'].cuda(local_rank,non_blocking=True)
        
        word_ids = samples['word_ids'].squeeze(1)
        word_ids = word_ids.cuda(local_rank,non_blocking=True)
        llm_word_ids = samples['llm_word_ids'].squeeze(1)
        llm_word_ids = llm_word_ids.cuda(local_rank,non_blocking=True)[:, :3, :]  # 选择前两个simple text
        
        diff_ref_word_ids_0 = samples['diff_ref_word_ids_0'].squeeze(1)
        diff_ref_word_ids_0 = diff_ref_word_ids_0.cuda(local_rank,non_blocking=True)
        diff_ref_llm_word_ids_0 = samples['diff_ref_llm_word_ids_0'].squeeze(1)
        diff_ref_llm_word_ids_0 = diff_ref_llm_word_ids_0.cuda(local_rank,non_blocking=True)[:, :3, :]  # 选择前两个simple text
        
        # diff_ref_word_ids_1 = samples['diff_ref_word_ids_1'].squeeze(1)
        # diff_ref_word_ids_1 = diff_ref_word_ids_1.cuda(local_rank,non_blocking=True)
        # diff_ref_llm_word_ids_1 = samples['diff_ref_llm_word_ids_1'].squeeze(1)
        # diff_ref_llm_word_ids_1 = diff_ref_llm_word_ids_1.cuda(local_rank,non_blocking=True)[:, :3, :]  # 选择前两个simple text
        
        all_word_ids = torch.cat([word_ids, diff_ref_word_ids_0], dim=0)
        all_llm_word_ids = torch.cat([llm_word_ids, diff_ref_llm_word_ids_0], dim=0)
        # category_word_ids = samples['category_word_id'].squeeze(1)
        # category_ids = samples['category_id']
        # category_word_ids = category_word_ids.cuda(local_rank,non_blocking=True)
        # category_ids = category_ids.cuda(local_rank,non_blocking=True)

        target = targets['target'].cuda(local_rank,non_blocking=True)
        bbox = targets['boxes'].cuda(local_rank,non_blocking=True) 
       

        B,c,h,w = img.shape
        raw_sentences = targets['sentences']
        
        with torch.cuda.amp.autocast(enabled=config.amp_training):
            if args.mode != 'clip':
                cls, _, _, sig_out, _ = model(img, raw_sentences)  
            else:
                cls_loss, fg_loss, cbs_loss, shrink_loss, sam_loss, diversify_loc_loss = model(img, all_word_ids, all_llm_word_ids, clip_model, samples, \
                                                                                               div_loc=True)  
                # cls_loss, fg_loss, cbs_loss, shrink_loss, sam_loss, diversify_loc_loss = model(img, word_ids, llm_word_ids, clip_model, samples, div_loc=False)  
        
        l1 = fg_loss 
        l4 = cls_loss 

        # l2 = inst_contra_loss if epoch > 0 else inst_contra_loss * 0.0
        l2 = shrink_loss
        l3 = sam_loss
        if args.negative_samples > 0:
            l5 = cbs_loss 
        else:
            l5 = torch.tensor(0)

        loss = l1 * args.w1 + l2 * args.w2 + l3 * args.w3 + l4 * args.w4 + l5 * args.w5 + diversify_loc_loss * args.div_loc

        loss = loss / args.gradient_accumulation_steps
        
        # Synchronizes all processes.
        # all process statistic
        # loss.backward()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        torch.cuda.synchronize()
        
        if local_rank == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('optim/lr', lr, iteration)
            writer.add_scalar('train/loss', loss.data.cpu().numpy(), iteration)
            writer.add_scalar('train/fg_loss', l1.data.cpu().numpy(), iteration)
            writer.add_scalar('train/shrink_loss', l2.data.cpu().numpy(), iteration)
            writer.add_scalar('train/anchor_loss', l3.data.cpu().numpy(), iteration)
            writer.add_scalar('train/cls_loss', l4.data.cpu().numpy(), iteration)
            writer.add_scalar('train/cbs_loss', l5.data.cpu().numpy(), iteration)
            writer.add_scalar('train/diversify_loc_loss', diversify_loc_loss.data.cpu().numpy(), iteration)
            
            if idx % 20 == 0:
                # print cpu and gpu info
                gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
                # logger.info("Current GPU info: Total {} MB, Used {} MB, Free {} MB" .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
                cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
                # logger.info("Current CPU info: Total {} MB, Used {} MB, Free {} MB" .format(cpu_mem_total, cpu_mem_process_used, cpu_mem_free))
                writer.add_scalar('device/GPU_Used', gpu_mem_used, iteration)
                writer.add_scalar('device/GPU_Free', gpu_mem_free, iteration)
                writer.add_scalar('device/CPU_Used', cpu_mem_process_used, iteration)
                writer.add_scalar('device/CPU_Free', cpu_mem_free, iteration)

        loss_meter.update(loss.item(), target.size(0))
        batch_time.update(time.time()-end)
        end=time.time()
        
        if idx % args.print_freq==0 and local_rank==0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas=batch_time.avg*(num_steps-idx)
            all_etas = batch_time.avg * (max_iter - iteration)
            logger.info(
                f'Train:[{epoch:2d}/{args.epoch}][{idx:4d}/{num_steps}] | '
                f'eta: {datetime.timedelta(seconds=int(etas))} | lr {lr:.6f} || '
                f'loss: {loss_meter.val:.3f} ({loss_meter.avg:.3f}) | '
                f'l1: {l1:.3f} | '
                f'l2: {l2.item():.3f} | '
                f'l3: {l3.item():.3f} | '
                f'l4: {l4:.3f} | '
                f'l5: {l5:.3f} | '
                # f'time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                # f'mem: {memory_used:.0f}MB || '
                f'all_eta: {datetime.timedelta(seconds=int(all_etas))}')
        iteration += 1
    epoch_time=time.time()-start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return iteration



if __name__=="__main__":
    parse=get_parser()
    args=parse.parse_args()

    print('========='*10)
    print(args)
    print('========='*10)

    if args.vis_out is not None and not os.path.exists(args.vis_out):
        os.mkdir(args.vis_out)

    # print(f'[{args.w1}, {args.w2}, {args.w3}, {args.w4}, {args.w5}]')
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank=int(os.environ['RANK'])
        world_size=int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank=-1
        world_size=-1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
        
    if args.distributed:
        logger = create_logger(dist_rank=dist.get_rank())
    else:
        logger = create_logger()
    
    global writer 
    writer = SummaryWriter(args.board_folder)

    # svae code
    import inspect
    import shutil
    os.makedirs(args.save_code, exist_ok=True)
    model_py_file = inspect.getfile(TRIS)

    shutil.copy(model_py_file, args.save_code)

    main(args) 


