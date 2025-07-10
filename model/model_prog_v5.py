import os
import random
# os.environ['CUDA_ENABLE_DEVICES'] = '0,1'

import numpy as np 
import torch 
import torch.nn as nn
# import transformers
import torch.nn.functional as F
from model.attn import bilateral_prompt

import CLIP.clip as clip 
from model.utils import Upsample, clip_forward, focal_loss, MaxLoss, get_norm_cam, SAM_shrink_loss_MSE, SAM_shrink_loss, SAM_diversify_loss


class TRIS(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args 
        self.bert_model = args.bert_tokenizer
        
        if args.backbone == 'clip-RN50':
            last_vis_channel = 2048 
            self.textdim = 1024
        elif args.backbone == 'clip-RN101':
            last_vis_channel = 2048 
            self.textdim = 512
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        type = args.backbone.split('-')[-1]
        clip_model, _ = clip.load(type, device=device, jit=False, txt_length=args.max_query_len)
        # clip_model, _ = clip.load('../clip_weights/RN50.pt', device=device, jit=False, txt_length=args.max_query_len)
        clip_model = clip_model.float() 

        if 'clip-RN' in args.backbone:
            self.backbone = clip_model 
            
        self.c4_vis_project = nn.Conv2d(in_channels=last_vis_channel, out_channels=args.hidden_dim, kernel_size=1, bias=True)
        # self.c3_vis_project = nn.Conv2d(in_channels=last_vis_channel//2, out_channels=args.hidden_dim, kernel_size=1, bias=True)
        self.global_lan_project = nn.Linear(self.textdim, args.hidden_dim)
        # self.local_lan_project = nn.Linear(self.textdim//2, args.hidden_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.args.attn_multi > 0:
            self.attn_fusion_vl = nn.ModuleList()
            self.attn_fusion_vl.append(bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim))
            self.attn_fusion_vl.append(bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim))
            self.attn_fusion_vl.append(bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim))
            self.attn_fusion_vl.append(bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim))

            # self.attn_fusion_0 = bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim) 
            # self.attn_fusion_1 = bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim) 
            # self.attn_fusion_2 = bilateral_prompt(args.hidden_dim, lan_chans=args.hidden_dim) 

    def trainable_parameters(self):
        # newly_add_params = [self.vis_project, self.lan_project, self.logit_scale]
        newly_add_params = [self.c4_vis_project, self.global_lan_project]
        # newly_add_params = [self.c3_vis_project, self.global_lan_project]
        try:
            newly_add_params.append(self.attn_fusion_vl)
        except:
            print('no attn fusion')
        # breakpoint()
        backbone = self.backbone
        return (list(backbone.parameters()), list(nn.ModuleList(newly_add_params).parameters()))

    def TRIS_Loss(self, score, vis_x):
        B, _, h_, w_ = vis_x.size()

        score_t = score.transpose(1, 2).reshape(B, -1, h_, w_)
        bg = torch.ones_like(score_t[:,:1])
        score_t = torch.cat([bg, score_t], 1)
        
        bs, c, h, w = score_t.size()

        masks = F.softmax(score_t*self.args.softmax_T, dim=1) # default
        # masks = F.sigmoid(score_t) 

        features = score_t.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1) 

        # classification loss
        cls_10 = features.mean(-1)  # [bs, bs+1]
        cls_11 = torch.max(features, dim=-1).values  # [bs, bs+1]
        cls_1 = cls_10 + cls_11 

        # # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), p=self.args.FOCAL_P, c=self.args.FOCAL_LAMBDA)

        # # adding the losses together
        cls_out = cls_1[:, 1:] + cls_2[:, 1:]
        
        # foreground stats
        masks_ = masks_[:, 1:]   # [bs, bs, hw]
        labels = torch.eye(bs).to(masks.device)
        cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)
        
        return cls_out, cls_fg

    def cbs_loss(self, img_x, sig_out, word_id=None, same_ref_word_ids=None, clip_model=None, samples=None, target=None):
        B, _, H, _ = img_x.size()
        clip_input_size = 224
        
        if img_x.shape[2] != clip_input_size:
            cam_224 = F.interpolate(sig_out, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
            img_224 = F.interpolate(img_x, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
            # gt_224 = F.interpolate(torch.Tensor(target).float(), (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
        else:
            cam_224 = sig_out 
            img_224 = img_x 
        
        # gt_fg_224_eval = []
        # gt_bg_224_eval = []
        fg_224_eval = []
        bg_224_eval = [] 
        for i in range(len(img_224)):
            fg_224_eval.append(cam_224[i] * img_224[i])
            bg_224_eval.append((1 - cam_224[i]) * img_224[i])
            # gt_fg_224_eval.append(gt_224[i] * img_224[i])
            # gt_bg_224_eval.append((1. - gt_224[i]) * img_224[i])
            
        fg_224_eval = torch.stack(fg_224_eval, dim=0)
        bg_224_eval = torch.stack(bg_224_eval, dim=0)
        # gt_fg_224_eval = torch.stack(gt_fg_224_eval, dim=0)
        # gt_bg_224_eval = torch.stack(gt_bg_224_eval, dim=0)


        # pseudo map clip_loss
        fg_loss = MaxLoss(clip_forward(clip_model, fg_224_eval, word_id, same_ref_word_ids))
        bg_loss = MaxLoss(clip_forward(clip_model, bg_224_eval, word_id, same_ref_word_ids))
        
        if self.args.clip_fg and self.args.clip_bg:
            clip_loss = fg_loss - bg_loss
        
        if self.args.clip_fg and not self.args.clip_bg:
                clip_loss = fg_loss

        if not self.args.clip_fg and  self.args.clip_bg:
            clip_loss = - bg_loss

        if not self.args.clip_fg and  not self.args.clip_bg:
            clip_loss = fg_loss * 0.0 + bg_loss * 0.0
        # fg_loss = fg_loss - bg_loss

        # gt map clip_loss
        # gt_fg_loss = MaxLoss(clip_forward(clip_model, gt_fg_224_eval, word_id, same_ref_word_ids))
        # gt_bg_loss = MaxLoss(clip_forward(clip_model, gt_bg_224_eval, word_id, same_ref_word_ids))


        if self.args.negative_samples > 0:
            neg_phrases = samples['neg_word_ids'].cuda()
            image_features = clip_model.encode_image(fg_224_eval)
            cbs_loss = torch.tensor(.0, requires_grad=True, device='cuda:0') 
            for i_ in range(B):
                _, text_features = clip_model.encode_text(neg_phrases[i_])
                image_feature = image_features[i_].reshape(1, -1)
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                neg_score = torch.matmul(image_feature, text_features.transpose(0,1))
                cbs_loss = cbs_loss + (-(torch.log(1 - neg_score)).mean())
            cbs_loss /= B 

        return clip_loss, cbs_loss
    
    def compute_cam_loss(self, fuse_v, fuse_l, img_x, vis_x, word_id=None, samples=None, clip_model=None, cam_loss=True):
        img_size = img_x.shape[2:]
        B, _, h_, w_ = vis_x.size()
        labels = torch.eye(B).cuda() 

        # calculate CAM score
        score = torch.bmm(fuse_v, fuse_l.transpose(1,2))   # [bs, hw, bs]
        score = self.logit_scale.exp() * score
        masks_out = []
        for i in range(B):
            masks_out.append(score[i,:,i].view(1, h_, w_))
        masks_out = torch.stack(masks_out, dim=0)
        seg_final_out = Upsample(masks_out, img_size)
        # relu_seg, sigmoid_seg = F.relu(seg_final_out), torch.sigmoid(seg_final_out)

        # calculate loss
        if cam_loss:
            cls_out, _ = self.TRIS_Loss(score, vis_x)
            c4_cls_loss = F.multilabel_soft_margin_loss(cls_out, labels)

            c4_fg_loss, c4_cbs_loss = self.cbs_loss(
                    img_x=img_x[:B//1], 
                    sig_out=torch.sigmoid(seg_final_out),
                    word_id=word_id, 
                    same_ref_word_ids=None, 
                    clip_model=clip_model, 
                    samples=samples, 
                    target=None
                )
            return F.relu(seg_final_out), torch.sigmoid(seg_final_out), c4_cls_loss, c4_fg_loss, c4_cbs_loss
        else:
            return F.relu(seg_final_out), torch.sigmoid(seg_final_out), None, None, None

    def refer_core(self, k, prior_vis, prior_main_lan, current_sub_lan=None, vis_x=None):
        B, _, h_, w_ = vis_x.size()

        norm_prior_vis = prior_vis / prior_vis.norm(dim=-1, keepdim=True)
        prior_main_lan = prior_main_lan / prior_main_lan.norm(dim=-1, keepdim=True)
        
        # obtain  contextual_sub_lan
        try:
            # breakpoint()
            current_sub_lan = current_sub_lan / current_sub_lan.norm(dim=-1, keepdim=True)
            contextual_sub_lan = self.attn_fusion_vl[k].forward_Qt(norm_prior_vis.permute(0, 2, 1).reshape(B, -1, h_, w_), current_sub_lan.unsqueeze(dim=1).transpose(1, 2)) # B x 1 x C
            contextual_sub_lan = contextual_sub_lan.transpose(0, 1).repeat(B, 1, 1)  # B x B x C
            # breakpoint()
            update_main_lan = self.attn_fusion_vl[k].forward_Qt_Kt(prior_main_lan.unsqueeze(dim=1), contextual_sub_lan)
            updated_main_lan = prior_main_lan + update_main_lan.squeeze(dim=1) * self.args.attn_multi_text
        except:
            breakpoint()

        # return prior_main_lan
        # updated_main_lan = prior_main_lan + current_sub_lan * self.args.attn_multi_text

        # cross-modal feature enhancing
        updated_main_lan_ = updated_main_lan / updated_main_lan.norm(dim=-1, keepdim=True)
        updated_main_lan_ = updated_main_lan_.unsqueeze(dim=0).repeat(B, 1, 1)
        res_lan = self.attn_fusion_vl[k].forward_Qt(norm_prior_vis.permute(0, 2, 1).reshape(B, -1, h_, w_), updated_main_lan_.transpose(1, 2))
        updated_prior_lan = updated_main_lan_ + res_lan * self.args.attn_multi_text

        res_vis = self.attn_fusion_vl[k].forward_Qv(norm_prior_vis.permute(0, 2, 1).reshape(B, -1, h_, w_), updated_main_lan_.transpose(1, 2))
        updated_prior_vis = norm_prior_vis + res_vis.flatten(2).transpose(1, 2) * self.args.attn_multi_vis


        return updated_main_lan, updated_prior_vis, updated_prior_lan


    def c4_foward(self, img_x, vis_x, ref_lan, ref_word=None, sub_ref_lan=None, sub_ref_word=None, samples=None, clip_model=None, div_loc=False):
        img_size = img_x.shape[2:]
        B, _, h_, w_ = vis_x.size()
        labels = torch.eye(B).cuda()  
        # project the same dimension
        vis = self.c4_vis_project(vis_x.float())  # [bs, 1024, 10, 10 ]
        ref_lan = self.global_lan_project(ref_lan)  # bs x 1024
        sub_ref_lan = self.global_lan_project(sub_ref_lan)
        
        raw_vis = vis.flatten(2).transpose(1,2)  # [bs, hw, 1024]
        # raw_ref_lan = ref_lan.unsqueeze(0).repeat(B, 1 ,1)     # [bs, bs, 1024]
        raw_ref_lan = ref_lan # [bs, 1024]

        raw_vis = raw_vis / raw_vis.norm(dim=-1, keepdim=True)
        raw_ref_lan = raw_ref_lan / raw_ref_lan.norm(dim=-1, keepdim=True)


        cls_loss = 0.0
        fg_loss = 0.0
        cbs_loss = 0.0
        diversify_loc_loss = torch.tensor(0.).cuda()
        shrink_loss = torch.tensor(0)
        anchor_loss = torch.tensor(0)
        
        seg_relu = []
        seg_sig = []
        left_match = []
        right_match = []

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  iteration stages # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        updated_vis, updated_main_lan = raw_vis, raw_ref_lan
        K = self.args.K_Iters
        sigmoid_seg = None
        # prior_mask = torch.zeros_like(raw_ref_lan).cuda()

        # 循环 sub_ref
        for k in range(K):
            # 
            # prepare the current sub-word
            if k == (K-1):
                current_lan, current_word = raw_ref_lan, ref_word  # # 结尾要加上 main_ref
            else:
                sub_ref_lan_k = sub_ref_lan[:, k, :]
                # sub_ref_lan_k = sub_ref_lan_k.unsqueeze(0).repeat(B, 1 ,1)     # [bs, bs, 1024]
                sub_ref_lan_k = sub_ref_lan_k / sub_ref_lan_k.norm(dim=-1, keepdim=True)

                current_lan, current_word = sub_ref_lan_k, sub_ref_word[:, k, :] if sub_ref_word is not None else None
            
            # update query
            updated_main_lan, updated_vis, updated_lan  = self.refer_core(k=k,  
                                                                    prior_vis=updated_vis, 
                                                                    prior_main_lan=updated_main_lan, 
                                                                    current_sub_lan=current_lan, 
                                                                    vis_x=vis_x
                                                                    )
            # print(updated_vis.shape, updated_main_lan.shape)
            
            if self.training:
                # breakpoint()
                # calculate CAM loss
                relu_seg, sigmoid_seg, cls_loss_, fg_loss_, cbs_loss_ = self.compute_cam_loss(
                                                                                    fuse_v=updated_vis[:], 
                                                                                    fuse_l=updated_lan[:, :B], 
                                                                                    img_x=img_x, 
                                                                                    vis_x=vis_x,
                                                                                    word_id=current_word[:B], 
                                                                                    samples=samples, 
                                                                                    clip_model=clip_model,
                                                                                    cam_loss=True
                                                                                    )
                if div_loc:
                    relu_seg_diff, sigmoid_seg_diff, _, _, _ = self.compute_cam_loss(
                                                                                        fuse_v=updated_vis[:], 
                                                                                        fuse_l=updated_lan[:, B:2*B], 
                                                                                        img_x=img_x, 
                                                                                        vis_x=vis_x,
                                                                                        word_id=current_word[B:2*B], 
                                                                                        samples=samples, 
                                                                                        clip_model=clip_model,
                                                                                        cam_loss=False
                                                                                        )
                    diversify_loc_loss_0 = SAM_diversify_loss(relu_seg, relu_seg_diff, samples, loss_mask=samples['same_img_mark_0'].cuda())

                    # relu_seg_diff, sigmoid_seg_diff, _, _, _ = self.compute_cam_loss(
                    #                                                                     fuse_v=updated_vis[:], 
                    #                                                                     fuse_l=updated_lan[:, 2*B:], 
                    #                                                                     img_x=img_x, 
                    #                                                                     vis_x=vis_x,
                    #                                                                     word_id=current_word[2*B:], 
                    #                                                                     samples=samples, 
                    #                                                                     clip_model=clip_model,
                    #                                                                     cam_loss=False
                    #                                                                     )
                    # diversify_loc_loss_1 = SAM_diversify_loss(relu_seg, relu_seg_diff, samples, loss_mask=samples['same_img_mark_1'].cuda())
                    
                    diversify_loc_loss += diversify_loc_loss_0
                   
                cls_loss += cls_loss_
                fg_loss += fg_loss_
                cbs_loss += cbs_loss_
                
                left_iou, right_iou = SAM_shrink_loss(relu_seg, samples)
                left_match.append(left_iou)
                right_match.append(right_iou)        
            else:
                relu_seg, sigmoid_seg, _, _, _ = self.compute_cam_loss(
                                                                                    fuse_v=updated_vis, 
                                                                                    fuse_l=updated_lan, 
                                                                                    img_x=img_x, 
                                                                                    vis_x=vis_x,
                                                                                    word_id=current_word, 
                                                                                    samples=samples, 
                                                                                    clip_model=clip_model,
                                                                                    cam_loss=False
                                                                                    )
            seg_relu.append(relu_seg)
            seg_sig.append(sigmoid_seg)


        if self.training:
            assert K < 5, print('K is should not large than 4')
            if K== 4:
                optimize_id = random.choice(range(3))
                shrink_loss = 1. - (right_match[optimize_id] - right_match[optimize_id+1]) #-(left_iou_last - left_iou_0)
                anchor_loss = ((1. -  left_match[0]) + (1. -  left_match[1]) + (1. -  left_match[2]) + (1. -  left_match[3])) / K
            if K == 3:
                optimize_id = random.choice(range(2))
                shrink_loss = 1. - (right_match[optimize_id] - right_match[optimize_id+1]) #-(left_iou_last - left_iou_0)
                anchor_loss = ((1. -  left_match[0]) + (1. -  left_match[1]) + (1. -  left_match[2])) / K
            if K == 2:
                shrink_loss = 1. - (right_match[0] - right_match[1]) #-(left_iou_last - left_iou_0)
                anchor_loss = ((1. -  left_match[0]) + (1. -  left_match[1])) / K
            if K==1:
                # breakpoint()
                anchor_loss = (1. -  left_match[0])

            return cls_loss/K, fg_loss/K, cbs_loss/K, shrink_loss, anchor_loss, diversify_loc_loss/K
        else:
            return seg_relu[-1], seg_sig[-1]

    

    def forward(self, img_x, word_id, llm_word_ids=None, clip_model=None, samples=None, div_loc=False): 
        img_size = img_x.shape[2:]
        B, _, H, _ = img_x.size()
        # assert B > 24, print('batch size < 24')
        labels = torch.eye(B).cuda()  
        
        c1, c2, c3, c4, _ = self.backbone.encode_image(img_x) # c4: [bs, 2048, 10, 10 ], c3: [3, 1024, 20, 20]
        _, hidden = self.backbone.encode_text(word_id)  # [3, 1024]
        
        # breakpoint()
        # llm_iters = llm_word_ids.shape[1]
        # llm_word_ids_ = llm_word_ids.reshape(B*llm_iters, -1)
        # _, llm_hidden = self.backbone.encode_text(llm_word_ids_)  # [3, 1024]
        # # llm_word_ids = llm_word_ids.reshape(B, llm_iters, -1)
        # llm_hidden = llm_hidden.reshape(B, llm_iters, 1024)
        if not div_loc:
            llm_iters = llm_word_ids.shape[1]
            llm_word_ids_ = llm_word_ids.reshape(B*llm_iters, -1)
            _, llm_hidden = self.backbone.encode_text(llm_word_ids_)  # [3, 1024]
            # llm_word_ids = llm_word_ids.reshape(B, llm_iters, -1)
            llm_hidden = llm_hidden.reshape(B, llm_iters, 1024)
        else:
            llm_iters = llm_word_ids.shape[1]
            llm_word_ids_ = llm_word_ids.reshape(2*B*llm_iters, -1)
            _, llm_hidden = self.backbone.encode_text(llm_word_ids_)  # [3, 1024]
            # llm_word_ids = llm_word_ids.reshape(2*B, llm_iters, -1)
            llm_hidden = llm_hidden.reshape(2*B, llm_iters, 1024)

        if self.training:
            # c4_foward
            c4_cls_loss, c4_fg_loss, c4_cbs_loss, shrink_loss, match_loss, diversify_loc_loss = self.c4_foward(
                img_x=img_x,
                vis_x=c4, 
                ref_lan=hidden, 
                ref_word=word_id,
                sub_ref_lan=llm_hidden,
                sub_ref_word=llm_word_ids,
                samples=samples, 
                clip_model=clip_model,
                div_loc=div_loc)

            return c4_cls_loss, c4_fg_loss, c4_cbs_loss, shrink_loss, match_loss, diversify_loc_loss
        
        else:
            seg_relu, seg_sig = self.c4_foward(img_x=img_x,
                                               vis_x=c4, 
                                               ref_lan=hidden, 
                                               sub_ref_lan=llm_hidden,
                                               samples=samples)

            return seg_relu
      


if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

    from args import get_parser 
    parse=get_parser()
    args=parse.parse_args()

    m = TRIS(args).cuda()
    x = torch.randn(4, 3, 320, 320)
    x = torch.tensor(x, dtype=torch.float32)
    word_id = torch.ones(4, 20)
    word_id = torch.tensor(word_id, dtype=torch.int64)
    att_mask = torch.ones(4, 20)

    output = m(x.cuda(), word_id.cuda(), att_mask.cuda())
    print('success !!!')
