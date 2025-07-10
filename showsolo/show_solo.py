import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())



import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import os
import torch.nn.functional as F


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.75]])
        img[m] = color_mask
    ax.imshow(img)


def show_sub_anns(anns, image):
    """
    Visualize more attention maps from 1-12 layer,for class token
    Note: attn_weights is of shape n_layer x n_batch x n_token x n_token
    """
    # left = 0.1  # the left side of the subplots of the figure
    # right = 0.9   # the right side of the subplots of the figure
    # bottom = 0.1  # the bottom of the subplots of the figure
    # top = 0.05     # the top of the subplots of the figure
    # wspace = 0.5  # the amount of width reserved for space between subplots,
    #               # expressed as a fraction of the average axis width
    # hspace = 0.5  # the amount of height reserved for space between subplots,
    #               # expressed as a fraction of the average axis height

    # sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    show_masks = []
    for ann in anns:
        if ann.sum() > 100:
            show_masks.append(ann)
    # show_masks = anns
    ncols = int(len(show_masks) ** (0.5)) + 1
    nrows = len(show_masks)//ncols + 1

    print(nrows, ncols, len(show_masks))
    # breakpoint()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16))
    #fig.tight_layout()
    mask_sum = 0
    # axes[0][0].imshow(image)
    for row in range(nrows):
        for col in range(0, ncols):
            if (row*ncols + col) == 0:
                axes[row][col].imshow(image)
                axes[row][col].axis('off')
            else:
                if (row*ncols + col) <= len(show_masks):
                    # print('area: ', show_masks[row*ncols + col]['area'])
                    bool_mask = show_masks[row*ncols + col -1]
                    bool_mask = F.interpolate(bool_mask.unsqueeze(dim=0).unsqueeze(dim=0), size=image.shape[:2], mode="bilinear")[0][0]
                    sam_mask = np.float32(bool_mask)
                    mask_sum += sam_mask.sum()
                    
                    # image = torch.Tensor(image)
                    sam_mask = cv2.applyColorMap(np.uint8(sam_mask * 255), cv2.COLORMAP_JET)[:, :, ::-1]
                    # print(image.shape, sam_mask.shape)
                    # img_and_sam_map = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, sam_mask, 0.7, 0)
                    # img_and_sam_map = cv2.addWeighted(np.uint8(image * 255), 0.3, sam_mask, 0.7, 0)
                    img_and_sam_map = cv2.addWeighted(image, 0.3, sam_mask, 0.7, 0)
                    # print(row, col)
                    # attn_l=cv2.resize(attn_l, (224, 224))
                    axes[row][col].imshow(img_and_sam_map)
                    axes[row][col].axis('off')
    #plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)
    # plt.show()

def show_solo(name):

    image = cv2.imread(os.path.join('/home/yzq/mnt/RIS/coco/train2014', '{}.jpg'.format(name)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = np.load(os.path.join('/home/yzq/mnt/RIS/coco/refer/refcocog/SOLO_Mask/SOLO_out','{}.npy'.format(name)))
    masks = np.load(os.path.join('/home/yzq/mnt/RIS/coco/refer/refcocog/SAM_Mask/SAM_Out','{}.npy'.format(name)))
    masks = nn.functional.interpolate(torch.Tensor(masks).unsqueeze(dim=0), (image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)[0]
    # masks = masks>0.1
    # sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

    sam_out_path = "/home/yzq/mnt/RIS/coco/refer/refcocog/SAM_Mask/google_val_IoU_29.1_Prec_32.2_Recall_57.7_top_2.npy"
    sam_out = np.load(sam_out_path, allow_pickle=True).item()

    for ref_id in sam_out.keys():
        if name in sam_out[ref_id]['sam_mask_path']:
            this_ref_id = ref_id
            break
    ref_sam_out = sam_out[this_ref_id]
    candidate_id = ref_sam_out['candidate_id'][0]
    masks = masks * torch.Tensor(candidate_id).unsqueeze(dim=-1).unsqueeze(dim=-1)
    
    print(masks.shape)

    plt.figure(figsize=(8,8))
    plt.imshow(image)
    # show_anns(masks)
    show_sub_anns(masks, image)
    plt.axis('off')
    
    plt.savefig('./solo_vis/{}.png'.format(name), bbox_inches = 'tight')
    plt.show() 

if __name__ == "__main__":
    show_solo(name='COCO_train2014_000000313963')