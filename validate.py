import os
import torch 
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torch.distributed as dist
from dataset.transform import get_transform
from args import get_parser
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.util import AverageMeter, load_checkpoint
import torchvision.transforms as transforms
import cv2
from PIL import Image
import pycocotools.mask as mask_utils 

import time 
from logger import create_logger
import datetime
import numpy  as np 
from utils.util import compute_mask_IU , compute_multi_thre_IU
import torch.nn as nn 
from tensorboardX import SummaryWriter
from utils.box_eval_utils import eval_box_iou, generate_bbox, eval_box_acc
import CLIP.clip as clip 
import json 

from dataset.ReferDataset import ReferDataset 

# from model.model_base import TRIS
# from model.model_prog_v4 import TRIS  
from model.model_prog_v5 import TRIS  
# from model.model_prog_v3 import TRIS 
# from model.model_stage1_v2 import TRIS
# from model.model_stage2 import TRIS
from demo import get_norm_cam


def main(args):
    if args.distributed:
        local_rank=dist.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    # build module
    model = TRIS(args)
    if args.distributed:
        model.cuda(local_rank)
    else:
        model.cuda() 

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    else:
        model=torch.nn.DataParallel(model)

    model_without_ddp=model.module
    num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {num_params}")
    # build dataset
    val_datasets = [] 
    for test_split in args.test_split.split(','):
        val_datasets.append(ReferDataset(refer_data_root=args.refer_data_root,
                                            dataset=args.dataset,
                                            splitBy=args.splitBy,
                                            bert_tokenizer=args.bert_tokenizer,
                                            split=test_split,
                                            size=args.size,
                                            image_transforms=get_transform(args.size, train=False),
                                            eval_mode=True,
                                            scales=args.scales,
                                            max_tokens=args.max_query_len,
                                            positive_samples=args.positive_samples,
                                            pseudo_path=args.pseudo_path))  ######## 1 for multitext inference, else with same with train datasets
    if args.distributed:
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(DistributedSampler(val_dataset, shuffle=False))
    else:
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(None)

    val_loaders = [] 
    for val_dataset, val_sampler in zip(val_datasets, val_samplers):
        val_loaders.append(DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=0,
                            pin_memory=True, 
                            sampler=val_sampler,
                            shuffle=False))
    
    if args.resume:
        if args.pretrain is not None:
            load_checkpoint(args, model_without_ddp, logger=logger)  #####
        if args.eval:
            # validate(args,val_loader,model,local_rank)
            st = time.time()
            val_acc, testA_acc, testB_acc = 0, 0, 0
            for i, val_loader in enumerate(val_loaders):
                if args.prms:
                    oIoU, mIoU, hit = validate_same_sentence(args, val_loader, model, local_rank, save_cam=args.save_cam)
                else:
                    oIoU, mIoU, hit = validate(args, val_loader, model, local_rank, save_cam=args.save_cam)
                if i == 0: val_acc = mIoU
                elif i == 1: testA_acc = mIoU
                else: testB_acc = mIoU
                print()
                print()
            print(f'val: {val_acc}, testA, {testA_acc}, testB: {testB_acc}')
            all_t = time.time() - st 
            print(f'Testing time:  {str(datetime.timedelta(seconds=int(all_t)))}')
            # return


def isCorrectHit(bbox_annot, heatmap, gt_mask=None):
    max_loc = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)

    hitm = 0 
    max_point_score = gt_mask[max_loc[0], max_loc[1]] + 1
    if max_point_score.max() == 2:
        hitm = 1 

    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1, max_loc, hitm
    return 0, max_loc, hitm 

def Hit_SAM_id(sam_mask, heatmap, clip_top_k):
    max_loc = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    _, sam_H, sam_w = sam_mask.shape
    # breakpoint()
    per_sam_mask_pred = (sam_mask * heatmap.unsqueeze(dim=0)).reshape(-1, sam_H*sam_w)
    per_sam_mask_max_V = per_sam_mask_pred.max(dim=-1)[0]
    max_V, max_sam_Id = per_sam_mask_max_V.topk(5, dim=0, largest=True, sorted=True)
    hit_id = []
    hit_id.append(max_sam_Id[0])
    for k, val in enumerate(max_V[1:]):
        if val > 1.9:
           hit_id.append(max_sam_Id[k+1]) 
    return hit_id, 1
    
    # heatmap_thre = (heatmap>0.99) * heatmap
    # _, sam_H, sam_w = sam_mask.shape
    # # breakpoint()
    # per_sam_heatmap = sam_mask * heatmap.unsqueeze(dim=0).reshape(-1, sam_H*sam_w)
    # # per_sam_heatsum = per_sam_heatmap.sum(dim=-1)
    # per_sam_heatsum = per_sam_heatmap.sum(dim=-1) / sam_mask.reshape(-1, sam_H*sam_w).sum(dim=-1)
    # max_V, max_sam_Id = per_sam_heatsum.topk(5, dim=0, largest=True, sorted=True)
    # hit_id = []
    # hit_id.append(max_sam_Id[0])
    # return hit_id, 1
    
    # hit_each_id = []
    
    # for id_k in clip_top_k:
    #     hit_each_id.append((sam_mask[id_k][max_loc[0], max_loc[1]] + 1).max())
    # if max(hit_each_id) > 1:
    #     hit_flag = 1
    #     max_value = max(hit_each_id)
    #     hit_id = clip_top_k[hit_each_id.index(max_value)]
    # else:
    #     hit_flag = -1
    #     hit_id = 1000

    return hit_id, hit_flag

def max_IoU_id(masks, target):
    assert (target.shape[-2:] == masks.shape[-2:])
    # I = np.sum(np.logical_and(masks, target))
    # U = np.sum(np.logical_or(masks, target))
    
    C, H, W = target.shape
    target = target.reshape(C, H*W)
    masks = masks.reshape(1, H*W)
    # I = torch.logical_and(masks, target).sum(dim=-1)
    # I = (target * masks).sum(dim=-1)
    # U = torch.logical_or(masks, target).sum(dim=-1) + 1e-5
    # U = (target + masks).sum(dim=-1) + 1e-5
    # IoU = I / U
    I = (target * masks).sum(dim=-1)
    # U = (target + masks).sum(dim=-1) + 1e-5
    U = target.sum(dim=-1) + 1e-5
    IoU = I / U
    # max_IoU, max_IoU_index = torch.max(IoU, 0)
    max_IoU, max_IoU_index = IoU.topk(1, dim=0, largest=True, sorted=True)

    return max_IoU_index


def get_scores(clip_model, fg_224_eval, word_id):
    image_features = clip_model.encode_image(fg_224_eval)  # [N1, C]
    _, text_features = clip_model.encode_text(word_id)  # [N2, C]
    # normalization
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits_per_image = (image_features @ text_features.t())  # [N1, N2]
    return logits_per_image 


def save_semantic_map(image, original_pred, sam_refine_pred, sam_clip_mask, raw_label, img_path, save_dir, name, point):
    
    inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    image = inv_normalize(image)
    image = transforms.functional.resize(image, list(raw_label.shape))

    # save raw image
    Image.fromarray(np.uint8(image.permute(1, 2, 0) * 255)).save(os.path.join(save_dir, img_path))
    
    img_path = img_path.split('.')[0]
    # save sam_refine pred
    sam_refine_pred = sam_refine_pred
    sam_refine_pred = cv2.applyColorMap(np.uint8(sam_refine_pred * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    sam_refine_pred_map = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, sam_refine_pred, 0.7, 0)
    
    # add marker
    sam_refine_pred_map = cv2.drawMarker(sam_refine_pred_map, (int(point[1]),int(point[0])), color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=20, thickness=4, line_type=4)
    # sam_refine_pred_map = cv2.xfeatures2d.drawStarShape(sam_refine_pred_map, (int(point[1]),int(point[0])), 10, (0, 255, 0), 3)
    pseduo_map_fname = '{}_R_{}.png'.format(img_path, name.replace('/', '_'))
    Image.fromarray(sam_refine_pred_map).save(os.path.join(save_dir, pseduo_map_fname))
    
    # save original pred map
    sam_clip_mask = sam_clip_mask
    # sam_clip_mask = cv2.applyColorMap(np.uint8(sam_clip_mask * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    # sam_clip_map = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, sam_clip_mask, 0.7, 0)
    # instance_map_fname = '{}_sam_clip_{}.png'.format(img_path,  name.replace('/', '_'))
    # Image.fromarray(sam_clip_map).save(os.path.join(save_dir, instance_map_fname))

    # save original pred map
    original_pred = original_pred
    original_pred = cv2.applyColorMap(np.uint8(original_pred * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    original_pred_map = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, original_pred, 0.7, 0)
    instance_map_fname = '{}_O_{}.png'.format(img_path,  name.replace('/', '_'))
    Image.fromarray(original_pred_map).save(os.path.join(save_dir, instance_map_fname))

    # save gt_map
    gt_map = raw_label.cpu().numpy()
    gt_map = cv2.applyColorMap(np.uint8(gt_map * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    img_and_gt_map = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, gt_map, 0.7, 0)
    
    gt_map_fname = '{}_G_{}.png'.format(img_path,  name.replace('/', '_'))
    # img_and_gt_map = cv2.circle(img=img_and_gt_map,  center = (int(point[1]),int(point[0])), radius=5, color=(0, 255, 0), thickness=-1)
    Image.fromarray(img_and_gt_map).save(os.path.join(save_dir, gt_map_fname))


# def save_semantic_map(image, original_pred, sam_refine_pred, sam_clip_mask, raw_label, img_path, save_dir, name, point):
    
#     inv_normalize = transforms.Normalize(
#             mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#             std=[1/0.229, 1/0.224, 1/0.225]
#         )
#     image = inv_normalize(image)
#     image = transforms.functional.resize(image, list(raw_label.shape))
#     image = np.uint8(image.permute(1, 2, 0) * 255)

#     # 0. save raw image
#     Image.fromarray(image).save(os.path.join(save_dir, img_path))
    
#     masked_image = image.copy()
#     img_path = img_path.split('.')[0]
    
#     # 1. save response map
#     # 对 response map 阈值化处理后可视化
#     thre_label = np.array(np.expand_dims(original_pred>0.1, axis=-1)).repeat(3, axis=2)
#     masked_image = np.where(thre_label, np.array([255,165,0], dtype='uint8'), masked_image)
#     masked_image = masked_image.astype(np.uint8)
#     masked_image = cv2.addWeighted(image, 0.2, masked_image, 0.5, 0)

#     instance_map_fname = '{}_O_{}.png'.format(img_path,  name.replace('/', '_'))
#     # Image.fromarray(masked_image).save('./test.png')
#     Image.fromarray(masked_image).save(os.path.join(save_dir, instance_map_fname))
    
#     # 2. save GT
#     gt_map = raw_label.cpu().numpy()
#     gt_map = cv2.applyColorMap(np.uint8(gt_map * 255), cv2.COLORMAP_JET)[:, :, ::-1]
#     img_and_gt_map = cv2.addWeighted(image, 0.3, gt_map, 0.7, 0)

#     gt_map_fname = '{}_G_{}.png'.format(img_path,  name.replace('/', '_'))
#     # img_and_gt_map = cv2.circle(img=img_and_gt_map,  center = (int(point[1]),int(point[0])), radius=5, color=(0, 255, 0), thickness=-1)
#     Image.fromarray(img_and_gt_map).save(os.path.join(save_dir, gt_map_fname))
    

    
def load_rle_mask(rle_path):
    rle_path = os.path.join("/home/yzq/data/coco/refer/sam_rle/filter", rle_path)
    
    def rle2mask(rle_dict):
        mask = mask_utils.decode(rle_dict) 
        return mask

    def mask2rle(mask):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) 
        
        return rle  

    sam_rle_mask = json.load(open(rle_path))

    # RLE 2 Mask
    N = len(sam_rle_mask)
    H, W = sam_rle_mask[0]['segmentation']['size']
    sam_binary_mask = np.zeros((40, H, W), dtype=np.uint8)
    for k in range(min(40, N)):
        sam_binary_mask[k] = rle2mask(sam_rle_mask[k]['segmentation'])
        
    return torch.unsqueeze(torch.Tensor(sam_binary_mask), dim=0)

@torch.no_grad()
def validate(args, data_loader, model, local_rank=0, visualize=False, logger=None, save_cam=False):
    num_steps = len(data_loader)
    model.eval()

    print('------------------------------')
    print('Starting validation without PRMS')
    print('------------------------------')

    if save_cam:
        os.makedirs(args.name_save_dir, exist_ok=True)
    if save_cam:
        os.makedirs(args.cam_save_dir, exist_ok=True)
    if save_cam:
        os.makedirs(args.save_train_pseudo_dir, exist_ok=True)
        
    batch_time=AverageMeter()
    mIOU_meter=AverageMeter()
    I_meter=AverageMeter()
    U_meter=AverageMeter() 
    box_mIOU_meter = AverageMeter()
    box_Acc_meter = AverageMeter()
    SAM_mIOU_meter = AverageMeter()
    SAM_I_meter=AverageMeter()
    SAM_U_meter=AverageMeter() 


    start = time.time()
    end=time.time()
    len_data_loader = 0 
    hit_acc = 0 
    hitmask_acc = 0  
    hit_index_dict = {}
    for idx,(samples, targets) in enumerate(data_loader):
        img_id = targets['img_path'].numpy()[0]
        # breakpoint()
        # category_word_ids = samples['category_word_id'].squeeze(1)
        # category_ids = samples['category_id']
        # if samples["category"][0] != "person":
        #     continue
        word_ids = samples['word_ids'].squeeze(1)
        word_ids = word_ids.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]

        llm_word_ids = samples['llm_word_ids'].squeeze(1)
        llm_word_ids = llm_word_ids.cuda(local_rank,non_blocking=True)[:, :3, :]

        img = samples['img'].cuda(local_rank,non_blocking=True) # [B,3,H,W]
        batch_size = img.size(0)
        target = targets['target'].cuda(local_rank,non_blocking=True) #[B,ori_H,ori_W]
        
        bbox = targets['boxes']#.cuda(local_rank,non_blocking=True) 
        sentences = targets['sentences']
        
        sam_masks = samples['zs_mask']
        clip_score = samples['clip_score']
        
        # load RLE mask
        # sam_masks = load_rle_mask(rle_path='COCO_train2014_%012d.json'%img_id)
        sam_masks = TF.resize(sam_masks, target.shape[-2:], interpolation=InterpolationMode.NEAREST)[0]

        o_H,o_W = target.shape[-2:]

        max_sam_IoU = 0
        max_sam_score = -1
        for j in range(word_ids.size(-1)):
            len_data_loader += 1
            o_H, o_W = target.shape[-2:]
            word_id = word_ids[:,:,j]
            
            # output = model(img, word_id)  # test TRIS_Base
            output = model(img, word_id, llm_word_ids[:, :, :, j], samples=samples)  # stage_1
            # output = model(img, word_id)  # stage_2
            pred = F.interpolate(output, (o_H, o_W), align_corners=True, mode='bilinear').squeeze(0)
            
            # 正则化 semantic_pred
            # if semantic_out is not None:
            #     semantic_pred = F.interpolate(semantic_out, (o_H, o_W), align_corners=True, mode='bilinear').squeeze(0)
            #     vis_semantic_pred = get_norm_cam(semantic_pred.detach().cpu())
            # else:
            #     vis_semantic_pred = None
            # pred : 1 x h x w
            
            vis_instance_pred = get_norm_cam(pred.detach().cpu()[0])
      
            # 正则化 pred
            pred /= F.adaptive_max_pool2d(pred, (1, 1)) + 1e-5
            pred = pred.squeeze(0)
            t_cam = pred.clone()
            pred = pred.gt(1e-9)

            target = target.squeeze(0).squeeze(0)     
            
            hit, max_loc, hitmask = isCorrectHit(bbox.numpy(), t_cam.cpu().numpy().astype(np.float32), target)
            hit_acc += hit ########
            hitmask_acc += hitmask
            
            #######
            clip_socre_ = clip_score[j].squeeze()
            top_id = torch.argmax(clip_socre_)
            # sam_hit_mask = sam_masks[top_id]

            sam_I, sam_U = compute_mask_IU(target, sam_masks.cuda())
            sam_IoU = sam_I*1.0/sam_U 
            gt_sam_id = torch.argmax(sam_IoU)
            sam_I, sam_U = compute_mask_IU(target, sam_masks[gt_sam_id].cuda())
            sam_IoU = sam_I*1.0/sam_U 

            sam_hit_mask = sam_masks[gt_sam_id]

            ####### sam_mask refine之前的 IoU
            I, U = compute_mask_IU(target, pred)
            IoU = I*1.0/U 

            ####### sam_mask refine之后的 IoU
            to_refine_pred = pred
            # 1. 如果当前预测与 CLIP_score 最大的 sam_mask 交集较大，则选择此 sam_mask 作为最终预测
            # sam_pred_I, sam_pred_U = compute_mask_IU(to_refine_pred, sam_hit_mask.cuda())
            # sam_pred_IoU = sam_pred_I*1.0/sam_pred_U 
            # if sam_pred_IoU > 0.1 :
            #     to_refine_pred = sam_hit_mask.cuda()
            # else:
            #     continue
            # 2. 选择与当前预测交并比最大的sam_mask作为最终预测
            # indices = torch.argsort(clip_socre_.reshape(1, -1), dim=1, descending=True)[0]
            # sort_sam_masks = sam_masks[indices]
            
            # max_IoU_index = compute_multi_thre_IU(t_cam.cuda(), sam_masks.cuda())
            # max_IoU_index = max_IoU_id(t_cam.cpu(), sort_sam_masks[:6].cpu())
            # try:
            #     to_refine_pred = sam_masks[max_IoU_index.reshape(-1)][0]
            # except:
            #     breakpoint()
            # refine_by_sam_I, refine_by_sam_U = compute_mask_IU(target, to_refine_pred.cuda())
            # sam_IoU = refine_by_sam_I*1.0/refine_by_sam_U 
            
            # 3. 选择 hit 点所在的sam_mask
            # breakpoint()
            clip_top_k = clip_score[j][0].topk(15, dim=0, largest=True, sorted=True)[1]
            hit_id, hit_flag = Hit_SAM_id(sam_masks.cpu(), t_cam.cpu(), clip_top_k.cpu())
            if hit_flag != -1:
                # breakpoint()
                hit_id = torch.Tensor(hit_id).long()
                to_refine_pred = sam_masks.cuda()[hit_id].sum(dim=0)
                # tmp_I, tmp_U = compute_mask_IU(pred, tmp)
                # tmp_IoU = tmp_I*1.0/tmp_U 
                # if tmp_IoU > 0.2:
                #     to_refine_pred = tmp
            sam_I, sam_U = compute_mask_IU(target, to_refine_pred)
            sam_IoU = sam_I*1.0/sam_U 
            
            # if sam_IoU > max_sam_IoU:
            #     max_sam_IoU = sam_IoU
            # if clip_score[j][0][hit_id] > max_sam_score:
            #     max_sam_score = clip_score[j][0][hit_id]
            #     max_sam_IoU = sam_IoU
            hit_index_dict[idx] = hit_id
            #######
            bbox_gen = generate_bbox(pred.cpu().numpy().astype(np.float64))
            bbox_hit = bbox_gen[0]
            for bb in bbox_gen:
                if bb[0] <= max_loc[1] <= bb[2] and bb[1] <= max_loc[0] <= bb[3]:
                    bbox_hit = bb 
            box_miou = eval_box_iou(torch.tensor(bbox_hit[0:4]).unsqueeze(0), bbox)
            box_accu = eval_box_acc(bbox_gen, bbox) ### !!!box_acc for all generated boxes 
            #######

            I_meter.update(I)
            SAM_I_meter.update(sam_I)
            U_meter.update(U)
            SAM_U_meter.update(sam_U)
            mIOU_meter.update(IoU, batch_size)
            SAM_mIOU_meter.update(sam_IoU, batch_size)

            box_mIOU_meter.update(box_miou, batch_size)
            box_Acc_meter.update(box_accu, batch_size)
            
            if args.cam_save_dir is not None and save_cam:
                # root = os.path.join(args.cam_save_dir, f'{idx}_{j}_{img_id}.npy')
                # np.save(root, t_cam.cpu().numpy())
                # breakpoint()
                # print(targets['img_path_full'][0], sentences[j])
                # if "COCO_train2014_000000223165" in targets['img_path_full'][0]:
                #     print(samples['ref_id'].item())
                if idx < 0:
                    # breakpoint()
                    vis_instance_pred[:, 320:] = 0
                    save_semantic_map(image=img[0].cpu(), \
                                    original_pred=vis_instance_pred, \
                                    sam_refine_pred=to_refine_pred.cpu(), sam_clip_mask=sam_hit_mask.cpu(), raw_label=target, \
                                    img_path=targets['img_path_full'][0], \
                                    save_dir=args.cam_save_dir, name=sentences[j][0], point=max_loc)
                    
            # if args.name_save_dir is not None and save_cam:
            #     cam_out_name.append(f'{idx}_{j}_{img_id}')
                
            # if args.save_train_pseudo_dir is not None:
            #     img_n = targets['img_path_full'][0].split('.')[0]
            #     ref_id = samples['ref_id'].item()
            #     sent_id = j
            #     np.save(os.path.join(args.save_train_pseudo_dir, img_n+'_{}_{}.npy'.format(ref_id, sent_id)), to_refine_pred.cpu().numpy())
            break

        batch_time.update(time.time()-end)
        end=time.time()

        if idx % (args.print_freq*4)==0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas=batch_time.avg*(num_steps-idx)
            print(
                f'Test: [{idx:4d}/{num_steps}] | '
                f'mIOU {100*mIOU_meter.avg:.3f} | '
                f'Overall IOU {100*float(I_meter.sum)/float(U_meter.sum):.3f} | '
                f'Hit {hit_acc/len_data_loader*100:.3f} | '
                f'HitM {hitmask_acc/len_data_loader*100:.3f} | '
                f'SAM mIOU {100*SAM_mIOU_meter.avg:.3f} | '
                f'SAM Overall IOU {100*float(SAM_I_meter.sum)/float(SAM_U_meter.sum):.3f} | '
                f'eta: {datetime.timedelta(seconds=int(etas))} || '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})', flush=True)
    
    # if args.name_save_dir is not None and save_cam:
    #     with open(os.path.join(args.name_save_dir, f'{args.dataset}_train_cam_name.json'), 'w') as f:
    #         f.write(json.dumps(cam_out_name))
    
    overall_IoU = 100*float(I_meter.sum)/(float(U_meter.sum) + 0.001)
    mIOU = 100*mIOU_meter.avg
    hit = 100*hit_acc/len_data_loader 
    box_miou = 100*box_mIOU_meter.avg
    box_acc = 100*box_Acc_meter.avg
    SAM_mIOU = 100*SAM_mIOU_meter.avg
    SAM_oIOU = 100*float(SAM_I_meter.sum)/float(SAM_U_meter.sum)
    Hit_M = hitmask_acc/len_data_loader*100
    print(f'Test: mIOU {mIOU:.5f}  \
            Overall IOU {overall_IoU:.5f}  \
            HiT {100*hit_acc/len_data_loader:.3f}  \
            HitM {hitmask_acc/len_data_loader*100:.3f} \
            SAM mIOU {SAM_mIOU:.5f} \
            SAM oIOU {SAM_oIOU:.5f}')
    np.save('./hit_index_dict_1.npy', hit_index_dict)
    return overall_IoU, mIOU, hit, SAM_mIOU, SAM_oIOU, Hit_M


@torch.no_grad()
def validate_same_sentence(args, data_loader, model, local_rank=0, visualize=False, logger=None, save_cam=False):
    num_steps = len(data_loader)
    model.eval()

    print('------------------------------')
    print('Starting validation with PRMS')
    print('------------------------------')

    save_cam = args.save_cam 

    if save_cam and not os.path.exists(args.name_save_dir):
        os.makedirs(args.name_save_dir, exist_ok=True)
    if save_cam and not os.path.exists(args.cam_save_dir):
        os.makedirs(args.cam_save_dir, exist_ok=True)

    batch_time=AverageMeter()
    mIOU_meter=AverageMeter()
    I_meter=AverageMeter()
    U_meter=AverageMeter() 

    start = time.time()
    end=time.time()
    len_data_loader = 0 
    hit_acc = 0 
    hitmask_acc = 0

    clip_input_size = 224 
    ###############
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, txt_length=args.max_query_len)
    clip_model.eval()
    ###############
    cam_out_name = [] 
    for idx,(samples, targets) in enumerate(data_loader):
        # if (idx+1) % 100 == 0:
        #     break 

        img_id = targets['img_path'].numpy()[0]
        
        word_ids = samples['word_ids'].squeeze(1)
        word_masks = samples['word_masks'].squeeze(1)
        img = samples['img'].cuda(local_rank,non_blocking=True) # [B,3,H,W]
        batch_size = img.size(0)
        target = targets['target'].cuda(local_rank,non_blocking=True) #[B,ori_H,ori_W]
        word_ids = word_ids.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        word_masks = word_masks.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        bbox = targets['boxes'].cuda(local_rank,non_blocking=True) 
        sentences = targets['sentences']

        o_H,o_W = target.shape[-2:]

        img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
        max_info = {
            'score': -1,
            'index': -1,
            'cam': 0
        }

        for j in range(word_ids.size(-1)):
            len_data_loader += 1
            o_H, o_W = target.shape[-2:]
            word_id = word_ids[:,:,j]
            word_mask = word_masks[:,:,j]
            
            output = model(img, word_id)
            pred = F.interpolate(output, (o_H,o_W), align_corners=True, mode='bilinear').squeeze(0)

            cam_224 = F.interpolate(output, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
            fg_224_eval = []
            for i in range(len(img_224)):
                fg_224_eval.append(cam_224[i] * img_224[i])
            fg_224_eval = torch.stack(fg_224_eval, dim=0)            

            score = 0.
            for _i in range(word_ids.size(-1)):
                score += get_scores(clip_model, fg_224_eval, word_ids[:,:,_i]).item() 
            if score > max_info['score']:
                max_info['score'] = score
                max_info['index'] = j 
                max_info['cam'] = pred 
        
        pred = max_info['cam']

        pred /= F.adaptive_max_pool2d(pred, (1, 1)) + 1e-5
        pred = pred.squeeze(0)
        t_cam = pred.clone()
        pred = pred.gt(1e-9)
        target = target.squeeze(0).squeeze(0)     
        
        I, U = compute_mask_IU(target, pred)
        I = I*word_ids.size(-1)
        U = U*word_ids.size(-1)
        IoU = I*1.0/U 
        hit, max_loc, hitmask = isCorrectHit(bbox.cpu().numpy(), t_cam.cpu().numpy().astype(np.float32), target)
        hit_acc += hit * word_ids.size(-1) ########
        hitmask_acc += hitmask * word_ids.size(-1)

        I_meter.update(I, batch_size*word_ids.size(-1))
        U_meter.update(U, batch_size*word_ids.size(-1))
        mIOU_meter.update(IoU, batch_size*word_ids.size(-1))

        if args.cam_save_dir is not None and save_cam:
            root = os.path.join(args.cam_save_dir, f'{idx}_{img_id}.npy')
            # root = os.path.join(args.cam_save_dir, 'cam', f'{idx}_{img_id}.npy')
            np.save(root, t_cam.cpu().numpy())
        if args.name_save_dir is not None and save_cam:
            cam_out_name.append(f'{idx}_{img_id}')

        batch_time.update(time.time()-end)
        end=time.time()

        if idx % args.print_freq==0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas=batch_time.avg*(num_steps-idx)
            print(
                f'Test: [{idx:4d}/{num_steps}] | '
                f'mIOU {100*mIOU_meter.avg:.3f} | '
                f'Overall IOU {100*float(I_meter.sum)/float(U_meter.sum):.3f} | '
                f'Hit {hit_acc/len_data_loader*100:.3f} | '
                f'HitM {hitmask_acc/len_data_loader*100:.3f} | '
                f'eta: {datetime.timedelta(seconds=int(etas))} || '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})', flush=True)
    
    if args.name_save_dir is not None and save_cam:
        with open(os.path.join(args.name_save_dir, f'{args.dataset}_train_names.json'), 'w') as f:
            f.write(json.dumps(cam_out_name))

    overall_IoU = 100*float(I_meter.sum)/float(U_meter.sum)
    mIOU = 100*mIOU_meter.avg
    hit = 100*hit_acc/len_data_loader 
    print(f'Test: mIOU {mIOU:.5f}  \
            Overall IOU {overall_IoU:.5f}  \
            HiT {100*hit_acc/len_data_loader:.3f}  \
            HitM {hitmask_acc/len_data_loader*100:.3f}')
    return overall_IoU, mIOU, hit 




if __name__=="__main__":
    parse=get_parser()
    args=parse.parse_args()

    print('========='*10)
    # print(args)
    print('========='*10)

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
    if args.board_folder is not None:
        writer = SummaryWriter(args.board_folder)

    main(args) 


