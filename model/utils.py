import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)


def focal_loss(x, p = 1, c = 0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)


def clip_forward(clip_model, images, tokenized_text, same_ref_tokenized_text=None):
    image_features = clip_model.encode_image(images)
    _, text_features = clip_model.encode_text(tokenized_text)
    
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)
    N, C = text_features.size()
    text_features = text_features.reshape(N, C, 1)

    similarity = torch.matmul(image_features, text_features)

    if same_ref_tokenized_text is not None:
        _, same_ref_text_features = clip_model.encode_text(same_ref_tokenized_text)
        same_ref_text_features = same_ref_text_features / same_ref_text_features.norm(dim=-1, keepdim=True)
        same_ref_text_features = same_ref_text_features.reshape(N, C, 1)
        same_ref_similarity = torch.matmul(image_features, same_ref_text_features)
        similarity = similarity + same_ref_similarity
        
    return similarity

def MaxLoss(x):
    margin = 0
    weights = 1
    x = x.clamp(0.0001, 0.9999)
    return -(torch.log(x + margin) * weights).mean()

def get_norm_cam(cam):
    
    cam = torch.clamp(cam, min=0)
    cam_t = cam.flatten(2)
    cam_max = torch.max(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    cam_min = torch.min(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-5)
    # norm_cam = norm_cam.squeeze(0).squeeze(0).cpu().numpy()
    return norm_cam

# def dice_loss(masks, target):
#     assert (target.shape[-2:] == masks.shape[-2:])
#     # I = np.sum(np.logical_and(masks, target))
#     # U = np.sum(np.logical_or(masks, target))
#     B, _, H, W = target.shape
#     target = target.reshape(B, -1, H*W)
#     masks = masks.reshape(B, -1, H*W)
#     I = (target * masks).sum(dim=-1)
#     U = (target + masks).sum(dim=-1) + 1e-5
#     return I, U

def dice_loss(masks, target):
    assert (target.shape[-2:] == masks.shape[-2:])
    # I = np.sum(np.logical_and(masks, target))
    # U = np.sum(np.logical_or(masks, target))
    B, _, H, W = target.shape
    target = target.reshape(B, -1, H*W)
    masks = masks.reshape(B, -1, H*W)
    I = (target * masks).sum(dim=-1)
    U = (target + masks).sum(dim=-1) + 1e-5
    return I, U

def SAM_Match_loss(relu_cam, samples):

    # this_ref id
    zs_masks = samples['zs_mask'].cuda()     # bs x N x 32 x 32
    zs_clip_score = samples['clip_score'].cuda()     # bs x N x 32 x 32
    candidate_id = samples['candidate_id'].cuda()

    B, N, H, W = zs_masks.shape

    # normalize relu_cam
    pred = relu_cam / (F.adaptive_max_pool2d(relu_cam, (1, 1)) + 1e-5)
    pred = pred.squeeze(0)
    pred = F.relu(pred - 1e-5) # pred = pred.gt(1e-9).float(), 不可导！！！
    pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
    
    float_I, float_U = dice_loss(pred.float(), zs_masks)
    float_IoU = (float_I / (float_U + 1e-5)) * candidate_id
    # breakpoint()
    max_float_IoU = float_IoU.max(dim=-1)[0]
    SAM_Match_loss = 1.0 -  max_float_IoU.mean()

    # SAM_Match_loss 确保 relu_cam 的每个点都在 SAM_candidate_masks 中
    # this_ref_pred, diff_ref_pred = pred[:B], pred[B:]
    
    # right_candidate_mask = (sam_masks * right_candidate_mask.unsqueeze(dim=-1).unsqueeze(dim=-1))  # B x 60 x H x W
    # right_I, right_U = dice_loss(this_ref_pred, right_candidate_mask)   # B x 60
    # right_IoU = right_I / (right_U + 1e-5)
    # max_right_IoU = right_IoU.max(dim=-1)[0]
    # right_Match_loss = 1.0 -  max_right_IoU.mean()

    # left_candidate_mask = (sam_masks * left_candidate_mask.unsqueeze(dim=-1).unsqueeze(dim=-1))  # B x 60 x H x W
    # left_I, left_U = dice_loss(this_ref_pred, left_candidate_mask)   # B x 60
    # left_IoU = left_I / (left_U + 1e-5)
    # max_left_IoU = left_IoU.max(dim=-1)[0]
    # left_Match_loss = 1.0 -  max_left_IoU.mean()


    
    # SAM_Compact_loss 确保 relu_cam 的每个点都在某一个 SAM_candidate_masks 中
    # I_2, U_2 = dice_loss(this_ref_pred, sam_masks)   # B x 60
    # match_IoU_2 = I_2 / (U_2 + 1e-5)
    # max_IoU_ratio = match_IoU.max(dim=-1)[0] / (match_IoU_2.sum(dim=-1) + 1e-5)

    # SAM_Compact_loss = 1.0 - max_IoU_ratio.mean()

    # import torch.autograd as autograd
    # cam_grad = autograd.grad(SAM_Match_loss, relu_cam, retain_graph=True)[0]  # SAM_Match_loss.requires_grad
    if  SAM_Match_loss.requires_grad == False:
        breakpoint()
    # if  SAM_Compact_loss.requires_grad == False:
    #     breakpoint()
    
    return SAM_Match_loss


def SAM_shrink_loss(relu_cam, samples):

    # this_ref id
    zs_masks = samples['zs_mask'].cuda()     # bs x N x 32 x 32
    zs_clip_score = samples['clip_score'].cuda()     # bs x N x 32 x 32
    # candidate_id = samples['candidate_id'].cuda()
    gt_masks = samples['target'].cuda().float()

    B, N, H, W = zs_masks.shape

    # normalize relu_cam
    pred = relu_cam / (F.adaptive_max_pool2d(relu_cam, (1, 1)) + 1e-5)
    pred = pred.squeeze(0)
    pred = F.relu(pred - 1e-5) # pred = pred.gt(1e-9).float(), 不可导！！！
    pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)

    # gt_masks = F.interpolate(gt_masks, (H, W), mode='bilinear', align_corners=True)
    # breakpoint()
    # dice_loss(this_ref_pred, left_candidate_mask)
    # 对 candidate_id 对应的 mask 按照 IoU 或者 激活面积 或者 激活最大值 进行排序，
    # candidate_mask_max_Value = (pred * zs_masks).reshape(B, N, H*W).max(dim=-1)[0]  # B x N
    # indices = torch.argsort(candidate_mask_max_Value, dim=1, descending=True)    # B x N
    # avg_cam = (pred * zs_masks).reshape(B, N, H*W).sum(dim=-1) / (zs_masks.reshape(B, N, H*W).sum(dim=-1) + 1e-5)
    # indices = torch.argsort(avg_cam, dim=1, descending=True)    # B x N
    # avg_cam = avg_cam * candidate_id   # 仅对 candidate_id 进行排序

    # float_I, float_U = dice_loss(pred.float(), zs_masks)          # 请按照 gt_IoU 排序
    # float_IoU = (float_I / (float_U + 1e-5)) * candidate_id
    # indices = torch.argsort(float_IoU, dim=1, descending=True)

    indices = torch.argsort(zs_clip_score, dim=1, descending=True)  # 请按照 clip_score 排序

    # gt_I, gt_U = dice_loss(gt_masks.float(), zs_masks)
    # gt_IoU = gt_I / (gt_U + 1e-5)
    # indices = torch.argsort(gt_IoU, dim=1, descending=True)  # 请按照 gt_IoU 排序

    bool_I, bool_U = dice_loss(pred.gt(1e-9).float(), zs_masks)
    bool_IoU = bool_I / (bool_U + 1e-5)

    sorted_masks = []
    sorted_IoU = []
    for i in range(B):
        sorted_masks.append(zs_masks[i][indices[i]])
        sorted_IoU.append(bool_IoU[i][indices[i]])

    sorted_masks = torch.stack(sorted_masks)
    sorted_IoU = torch.stack(sorted_IoU)

    # candidate_num = candidate_id[0].sum()
    # if candidate_num % 2 ==0:
    #     left_candidate_mask = sorted_masks[:, :candidate_num//2]   # B x n/2 x 10 x 10
    #     right_candidate_mask = sorted_masks[:, candidate_num//2:candidate_num]  # B x n/2 x 10 x 10
    #     left_IoU_confidence = sorted_IoU[:, :candidate_num//2]
    #     right_IoU_confidence = sorted_IoU[:, candidate_num//2:candidate_num]
    # else:
    #     assert candidate_num == 1
    #     left_candidate_mask = sorted_masks[:, :candidate_num]   # B x n/2 x 10 x 10
    #     right_candidate_mask = sorted_masks[:, candidate_num:]  # B x n/2 x 10 x 10
    #     left_IoU_confidence = sorted_IoU[:, :candidate_num]
    #     right_IoU_confidence = sorted_IoU[:, candidate_num:candidate_num]

    left_candidate_mask = sorted_masks[:, :1]   # B x n/2 x 10 x 10
    right_candidate_mask = sorted_masks[:, 1:]  # B x n/2 x 10 x 10
    left_IoU_confidence = sorted_IoU[:, :1]
    right_IoU_confidence = sorted_IoU[:, 1:]
    
    # SAM_Match_loss 确保 relu_cam 的每个点都在 SAM_candidate_masks 中
    this_ref_pred, diff_ref_pred = pred[:B], pred[B:]
    
    if left_candidate_mask.shape[1] > 0:
        # left_candidate_mask = (sam_masks * left_candidate_mask.unsqueeze(dim=-1).unsqueeze(dim=-1))  # B x 60 x H x W
        left_I, left_U = dice_loss(this_ref_pred, left_candidate_mask)   # B x 60
        left_IoU = left_I / (left_U + 1e-5)
        max_left_IoU, max_left_idx = left_IoU.max(dim=-1)

        gt_left_I, gt_left_U = dice_loss(pred.gt(1e-9).float(), left_candidate_mask)   # B x 60
        gt_left_IoU = gt_left_I / (gt_left_U + 1e-5)
        gt_max_left_IoU, _ = gt_left_IoU.max(dim=-1)
    else:
        max_left_IoU = torch.zeros(B).cuda()


    if right_candidate_mask.shape[1] > 0:
        # right_candidate_mask = (sam_masks * right_candidate_mask.unsqueeze(dim=-1).unsqueeze(dim=-1))  # B x 60 x H x W
        right_I, right_U = dice_loss(this_ref_pred, right_candidate_mask)   # B x 60
        right_IoU = right_I / (right_U + 1e-5)
        max_right_IoU, max_right_idx = right_IoU.max(dim=-1)

        gt_right_I, gt_right_U = dice_loss(pred.gt(1e-9).float(), right_candidate_mask)   # B x 60
        gt_right_IoU = gt_right_I / (gt_right_U + 1e-5)
        gt_max_right_IoU, _ = gt_right_IoU.max(dim=-1)
    else:
        max_right_IoU = torch.zeros(B).cuda()

    if  max_right_IoU.requires_grad == False:
        breakpoint()
    
    left_conf_mask = (gt_max_left_IoU) > 0.0
    right_conf_mask = (gt_max_right_IoU) < 1.0
    # return max_left_IoU.mean(), max_right_IoU.mean()
    return (max_left_IoU*left_conf_mask).sum()/(left_conf_mask.sum()+0.0001), (max_right_IoU*right_conf_mask).sum()/(right_conf_mask.sum()+0.0001)


import torch.nn.functional as F
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    # U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())*0.0
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def Strainght_through_Gumbel(logits):
    
    # sample = torch.rand_like(logits, requires_grad=True)
    argmax = torch.zeros_like(logits)
    argmax_one_hot = argmax.scatter(-1, torch.argmax(logits, dim=-1).unsqueeze(dim=-1), 1)
    
    C = argmax_one_hot - torch.softmax(logits, dim=-1)
    C = C.detach()
    
    SG_logits = C + torch.softmax(logits, dim=-1)
    
    return SG_logits

def SAM_diversify_loss(relu_cam, relu_cam_diff, samples, loss_mask):
    
    # this_ref id
    zs_masks = samples['zs_mask'].cuda()     # bs x N x 32 x 32
    # zs_clip_score = samples['clip_score'].cuda()     # bs x N x 32 x 32
    # candidate_id = samples['candidate_id'].cuda()
    # gt_masks = samples['target'].cuda().float()

    B, N, H, W = zs_masks.shape

    # normalize relu_cam
    pred = relu_cam / (F.adaptive_max_pool2d(relu_cam, (1, 1)) + 1e-5)
    pred = pred.squeeze(0)
    pred = F.relu(pred - 1e-5) # pred = pred.gt(1e-9).float(), 不可导！！！
    pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
    
    diff_pred = relu_cam_diff / (F.adaptive_max_pool2d(relu_cam_diff, (1, 1)) + 1e-5)
    diff_pred = diff_pred.squeeze(0)
    diff_pred = F.relu(diff_pred - 1e-5) # pred = pred.gt(1e-9).float(), 不可导！！！
    diff_pred = F.interpolate(diff_pred, (H, W), mode='bilinear', align_corners=True)

    pred_sam = pred * zs_masks
    pred_sam_max = pred_sam.reshape(B, N, H*W).max(dim=-1)[0]   # B x N , maxpooling
    # pred_sam_max = pred_sam.reshape(B, N, H*W).sum(dim=-1) / (zs_masks.reshape(B, N, H*W).sum(dim=-1) + 0.0001) # B x N , avgpooling
    
    diff_pred_sam = diff_pred * zs_masks
    diff_pred_sam_max = diff_pred_sam.reshape(B, N, H*W).max(dim=-1)[0]  # B x N , maxpooling
    # diff_pred_sam_max = diff_pred_sam.reshape(B, N, H*W).sum(dim=-1) / (zs_masks.reshape(B, N, H*W).sum(dim=-1) + 0.0001) # B x N , avgpooling
    
    # diff_pred_cat = torch.cat([pred_sam_max.unsqueeze(dim=1), diff_pred_sam_max.unsqueeze(dim=1)], dim=1)  # B x 2 x N
    # diff_pred_cat_max = torch.max(diff_pred_cat, dim=1)[0]
    # pooling_max_sum = torch.topk(diff_pred_cat_max, 2, dim=-1, largest=True, sorted=True)[0].sum(dim=-1)
    
    # loss =  (pred_sam_max.max(dim=-1)[0] + diff_pred_sam_max.max(dim=-1)[0]) - pooling_max_sum
    # loss = loss.mean()
    
    pred_sam_max_one_hot = Strainght_through_Gumbel(pred_sam_max)
    diff_pred_sam_max_one_hot = Strainght_through_Gumbel(diff_pred_sam_max)
    
    same_img_mark = loss_mask
    div_loss = 1.0 -  (torch.abs(pred_sam_max_one_hot - diff_pred_sam_max_one_hot).sum(dim=-1) * same_img_mark).sum() / (2*same_img_mark.sum()+0.0001)
    
    if  div_loss.requires_grad == False:
        breakpoint()
        
    return div_loss



# def SAM_diversify_loss(relu_cam, relu_cam_diff, samples, loss_mask):
    
#     # this_ref id
#     zs_masks = samples['zs_mask'].cuda()     # bs x N x 32 x 32
#     # zs_clip_score = samples['clip_score'].cuda()     # bs x N x 32 x 32
#     # candidate_id = samples['candidate_id'].cuda()
#     # gt_masks = samples['target'].cuda().float()

#     B, N, H, W = zs_masks.shape

#     # normalize relu_cam
#     pred = relu_cam / (F.adaptive_max_pool2d(relu_cam, (1, 1)) + 1e-5)
#     pred = pred.squeeze(0)
#     pred = F.relu(pred - 1e-5) # pred = pred.gt(1e-9).float(), 不可导！！！
#     pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
    
#     diff_pred = relu_cam_diff / (F.adaptive_max_pool2d(relu_cam_diff, (1, 1)) + 1e-5)
#     diff_pred = diff_pred.squeeze(0)
#     diff_pred = F.relu(diff_pred - 1e-5) # pred = pred.gt(1e-9).float(), 不可导！！！
#     diff_pred = F.interpolate(diff_pred, (H, W), mode='bilinear', align_corners=True)

#     pred_sam = pred * zs_masks
#     pred_sam_max = pred_sam.reshape(B, N, H*W).max(dim=-1)[0]   # B x N , maxpooling
#     # pred_sam_max = pred_sam.reshape(B, N, H*W).sum(dim=-1) / (zs_masks.reshape(B, N, H*W).sum(dim=-1) + 0.0001) # B x N , avgpooling
    
#     diff_pred_sam = diff_pred * zs_masks
#     diff_pred_sam_max = diff_pred_sam.reshape(B, N, H*W).max(dim=-1)[0]  # B x N , maxpooling
#     # diff_pred_sam_max = diff_pred_sam.reshape(B, N, H*W).sum(dim=-1) / (zs_masks.reshape(B, N, H*W).sum(dim=-1) + 0.0001) # B x N , avgpooling
    
#     # KL loss
    
    
#     def kl_divergence(p, q):
#         """
#         计算两组概率分布之间的 KL 散度。

#         参数：
#             p: 第一个概率分布张量 (shape=(n_samples, n_features))。
#             q: 第二个概率分布张量 (shape=(n_samples, n_features))。

#         返回值：
#             每对概率分布的 KL 散度 (shape=(n_samples,))。
#         """
#         return torch.sum(torch.where(p != 0, p * torch.log2(p / q), torch.tensor(0.).cuda()), dim=1)

#     def js_divergence(p, q):
#         """
#         计算两组概率分布之间的 JS 散度。

#         参数：
#             p: 第一个概率分布张量 (shape=(n_samples, n_features))。
#             q: 第二个概率分布张量 (shape=(n_samples, n_features))。

#         返回值：
#             每对概率分布的 JS 散度 (shape=(n_samples,))。
#         """
#         m = 0.5 * (p + q)
#         return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

#     # 示例用法
#     p = pred_sam_max
#     q = diff_pred_sam_max

#     # 归一化，确保每行是有效的概率分布
#     # p = p / (torch.sum(p, dim=1, keepdim=True) + 0.0001)
#     # q = q / (torch.sum(q, dim=1, keepdim=True) + 0.0001)
#     p = F.softmax(p, dim=-1)
#     q = F.softmax(q, dim=-1)

#     js = js_divergence(p, q)
    
#     if  js.requires_grad == False:
#         breakpoint()
#     # print(1.0 - js.mean())
#     return 1.0 - js.mean()
    

def SAM_shrink_loss_MSE(relu_cam, samples):

    # this_ref id
    zs_masks = samples['zs_mask'].cuda()     # bs x N x 32 x 32
    zs_clip_score = samples['clip_score'].cuda()     # bs x N x 32 x 32
    candidate_id = samples['candidate_id'].cuda()
    gt_masks = samples['target'].cuda().float()

    B, N, H, W = zs_masks.shape

    # normalize relu_cam
    pred = relu_cam / (F.adaptive_max_pool2d(relu_cam, (1, 1)) + 1e-5)
    pred = pred.squeeze(0)
    pred = F.relu(pred - 1e-5) # pred = pred.gt(1e-9).float(), 不可导！！！
    pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)

    gt_masks = F.interpolate(gt_masks, (H, W), mode='bilinear', align_corners=True)
    
    # dice_loss(this_ref_pred, left_candidate_mask)
    # 对 candidate_id 对应的 mask 按照 IoU 或者 激活面积 或者 激活最大值 进行排序，
    # candidate_mask_max_Value = (pred * zs_masks).reshape(B, N, H*W).max(dim=-1)  # B x N
    # indices = torch.argsort(candidate_mask_max_Value, dim=1, descending=True)    # B x N
    # avg_cam = (pred * zs_masks).reshape(B, N, H*W).sum(dim=-1) / (zs_masks.reshape(B, N, H*W).sum(dim=-1) + 1e-5)
    # avg_cam = avg_cam * candidate_id   # 仅对 candidate_id 进行排序
    float_I, float_U = dice_loss(pred.float(), zs_masks)
    float_IoU = (float_I / (float_U + 1e-5)) * candidate_id
    indices = torch.argsort(float_IoU, dim=1, descending=True)

    bool_I, bool_U = dice_loss(pred.gt(1e-9).float(), zs_masks)
    bool_IoU = bool_I / (bool_U + 1e-5)

    sorted_masks = []
    sorted_IoU = []
    for i in range(B):
        sorted_masks.append(zs_masks[i][indices[i]])
        sorted_IoU.append(bool_IoU[i][indices[i]])

    sorted_masks = torch.stack(sorted_masks)
    sorted_IoU = torch.stack(sorted_IoU)

    
    left_candidate_mask = sorted_masks[:, :1]   # B x n/2 x 10 x 10
    right_candidate_mask = sorted_masks[:, 1:2]  # B x n/2 x 10 x 10
    left_IoU_confidence = sorted_IoU[:, :1]
    right_IoU_confidence = sorted_IoU[:, 1:2]
    
    # SAM_Match_loss 确保 relu_cam 的每个点都在 SAM_candidate_masks 中
    this_ref_pred, diff_ref_pred = pred[:B], pred[B:]
    
    if right_candidate_mask.shape[1] > 0:
        right_mean = (this_ref_pred * right_candidate_mask).reshape(B, -1, H*W).sum(dim=-1) / (right_candidate_mask.reshape(B, -1, H*W).sum(dim=-1) + 1e-5)
        max_right_mean, max_right_idx = right_mean.max(dim=-1)
        # max_right_one_hot = torch.zeros(B, right_candidate_mask.shape[1]).cuda().scatter_(1, max_right_idx.reshape(-1, 1), 1)
        # right_conf = (right_IoU_confidence * max_right_one_hot).sum(dim=-1) > 0.1
        # max_right_IoU = max_right_IoU.mean()
        gt_right_I, gt_right_U = dice_loss(gt_masks, right_candidate_mask)   # B x 60
        gt_right_IoU = gt_right_I / (gt_right_U + 1e-5)
        gt_max_right_IoU, _ = gt_right_IoU.max(dim=-1)
    else:
        max_right_IoU = torch.zeros(B).cuda()
    
    if left_candidate_mask.shape[1] > 0:
        
        left_candidate_mask_pred = (this_ref_pred * left_candidate_mask)  # B x 60 x H x W
        left_I, left_U = dice_loss(left_candidate_mask_pred, left_candidate_mask)   # B x 60
        left_IoU = left_I / (left_U + 1e-5)
        max_left_IoU, max_left_idx = left_IoU.max(dim=-1)
        # max_left_one_hot = torch.zeros(B, left_candidate_mask.shape[1]).cuda().scatter_(1, max_left_idx.reshape(-1, 1), 1)
        # left_conf = (left_IoU_confidence * max_left_one_hot).sum(dim=-1) > 0.1
        # max_left_IoU = max_left_IoU.mean()
        gt_left_I, gt_left_U = dice_loss(pred.gt(1e-9).float(), left_candidate_mask)   # B x 60
        gt_left_IoU = gt_left_I / (gt_left_U + 1e-5)
        gt_max_left_IoU, _ = gt_left_IoU.max(dim=-1)
    else:
        max_left_IoU = torch.zeros(B).cuda()

    # zs_shrink_loss = -1 * (max_left_IoU - max_right_IoU*0.).mean()
    # zs_shrink_loss = (max_left_IoU - max_right_IoU).mean()
    # zs_shrink_loss = max_left_IoU.mean()
    # import torch.autograd as autograd
    # cam_grad = autograd.grad(SAM_Match_loss, relu_cam, retain_graph=True)[0]  # SAM_Match_loss.requires_grad
    if  max_left_IoU.requires_grad == False:
        breakpoint()
    
    conf_mask = (gt_max_left_IoU) > -1
    # return max_left_IoU.mean(), max_right_IoU.mean()
    return max_left_IoU.mean(), max_right_mean.mean()
    # return (max_left_IoU*conf_mask).sum()/(conf_mask.sum()+0.0001), (max_right_IoU*conf_mask).sum()/(conf_mask.sum()+0.0001)
    