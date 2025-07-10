from inspect import getcallargs
import os 
import sys
from tkinter.tix import Tree
from traceback import print_tb
# from dataset.RefTR_Dataset import denorm 
import torch.utils.data as data
import torch 
import numpy as np 
from PIL import Image 
import cv2 
import transformers
from dataset.refer import REFER
import CLIP.clip as clip 
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import pdb 
import imageio

# llm_out_dir = "/home/yzq/mnt/RIS/coco/refer/refcocog/Mistral_7B_Ref_Sents"
zs_sam_out_dir = "/home/yzq/mnt/RIS/coco/refer/refcocog/SAM_Mask/SAM_Out"
zs_solo_out_dir = "/home/yzq/mnt/RIS/coco/refer/refcocog/SOLO_Mask/SOLO_out"

def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)



def llm_out(json_file):
    import json
    
    llm_sents = []

    with open(json_file, 'r') as llm_file:
        llm_data = json.load(llm_file)
    
    llm_sents = []
    max_len = 0
    
    for sent_id in llm_data.keys():
        sents = llm_data[sent_id][:5]
        llm_sents.append(sents)
        # if len(sents.split(' ')) > max_len:
        #     max_len = len(sents.split(' '))

    llm_file.close()
    # if max_len>25:
    #     print(llm_sents)
        # if os.path.exists(json_file): 
        #     # 删除文件
        #     os.remove(json_file) 
        #     print('remove {} done'.format(json_file))
    return llm_sents

def sam_out(refer_data_root, dataset, splitBy, split, zs_model='SAM_Mask'):

    if dataset == 'refcocog':
        sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_candidate_2_4_6_8.npy'.format(splitBy))
        # if split == 'val':
        #     sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_mIoU_34.8_oIoU_30.8_candidate_2_4_6_8.npy'.format(splitBy, split))
        #     # sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_IoU_29.1_Prec_32.2_Recall_57.7_top_2.npy'.format(splitBy, split))
        #     # sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_IoU_38.1_Prec_44.2_Recall_43.5_top_1.npy'.format(splitBy, split))
        #     # sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_mIoU_34.8_oIoU_30.8_candidate_2_4_6_8.npy'.format(splitBy, split))
        # else:
        #     sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_mIoU_34.8_oIoU_30.9_candidate_2_4_6_8.npy'.format(splitBy, split))
        #     # sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_IoU_25.1_Prec_27.5_Recall_51.6_top_2.npy'.format(splitBy, split))
        #     # sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_IoU_34.3_Prec_39.3_Recall_39.8_top_1.npy'.format(splitBy, split))
        #     # sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_IoU_53.6_Prec_59.4_Recall_84.4_top_1+1.npy'.format(splitBy, split))
        #     # sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_{}_mIoU_34.8_oIoU_30.9_candidate_2_4_6_8.npy'.format(splitBy, split))
    elif dataset == "refcoco":
        sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_candidate_2_4_6_8.npy'.format(splitBy))
    else:
        sam_out_path = os.path.join(refer_data_root, 'refer', dataset, zs_model, '{}_candidate_2_4_6_8.npy'.format(splitBy))
    sam_out = np.load(sam_out_path, allow_pickle=True).item()

    return sam_out


class ReferDataset(data.Dataset):
    def __init__(self,
                 refer_data_root='/home/yzq/mnt/data/coco',
                 dataset='refcoco',
                 splitBy='unc',
                 bert_tokenizer='clip',
                 image_transforms=None,
                 max_tokens=20, 
                 split='train',
                 eval_mode=True,
                 size=448,
                 scales=False,
                 negative_samples=0,
                 positive_samples=1,
                 pseudo_path=None) -> None:
        """
        parameters:
            args: argparse obj
            image_transforms: transforms apply to image and mask
            max_tokens: determined the max length of token 
            split: ['train','val','testA','testB']
            eval_mode: whether in training or evaluating 
        """

        self.clip = ('clip' in bert_tokenizer)
        self.negative_samples = negative_samples
        self.positive_samples = positive_samples 
        self.classes=[]
        self.image_transforms=image_transforms
        self.split=split
        self.refer=REFER(refer_data_root, dataset, splitBy)
        self.scales = scales 
        self.size = size 
        self.pseudo_path = pseudo_path
        self.dataset = dataset
        self.refer_data_root = refer_data_root
        print('\nPreparing dataset .....')
        print(dataset, split)
        print(refer_data_root, dataset, splitBy) 
        print(f'pseudo_path = {pseudo_path}')

        self.max_tokens=max_tokens
        
        ref_ids=self.refer.getRefIds(split=self.split)
        zs_model = "SAM_Mask"
        self.sam_out = sam_out(refer_data_root, dataset, splitBy, split, zs_model)
        self.llm_out_dir = "/home/yzq/mnt/data/coco/refer/{}/Mistral_7B_Ref_Sents".format(dataset)
        self.zs_out_dir = "/home/yzq/mnt/data/coco/refer/{}/{}/{}_out".format(dataset, zs_model, zs_model.split('_')[0])
        

        if 'train' in split:
            # ref_ids = [28090, 42370, 21291,19004, 30585, 14190, 35610, 35140, 41970, 42521, 30290]
            ref_ids = ref_ids[:]
            print('\nNote: For faster training, we choose {} ref_ids of the {} data.\n'.format(len(ref_ids), split) * 2)
        else:
            # if "refcoco" in dataset and len(ref_ids) > 3000:
            all_len = len(ref_ids)
            ref_ids = ref_ids[:]
            print('\nNote: For faster evaluation, we choose {} ref_ids of the {} data.\n'.format(len(ref_ids), split) * 2)
        

        img_ids=self.refer.getImgIds(ref_ids)
        # change dict to list
        all_imgs=self.refer.Imgs
        self.imgs=list(all_imgs[i] for i in img_ids)
        
        self.ref_ids=ref_ids
        self.tokenizer = clip.tokenize 
        
        self.eval_mode = eval_mode

        self.input_ids=[]
        self.word_masks=[]
        self.all_sentences = []
        self.llm_input_ids=[]
        self.llm_word_masks=[]
        self.llm_all_sentences = []

        self.semantic_word_ids = []
        self.semantic_coco_id = []

        # get negative samples, 
        self.refid2index = {}
        
        for index, r in enumerate(self.ref_ids):
            self.refid2index[r] = index 
            # for each image
            ref=self.refer.Refs[r]
            # List[Tensor] Tensor shape [1,len]
            sentences_for_ref=[]
            attentions_for_ref=[]
            sentence_raw_for_re = []

            # for each sentence
            for i,(el,sent_id) in enumerate(zip(ref['sentences'],ref['sent_ids'])):
                sentence_raw = el['sent']
                
                word_id = self.tokenizer(sentence_raw).squeeze(0)[:self.max_tokens]
                word_id = np.array(word_id)
                word_mask = np.array(word_id>0,dtype=int)

                sentences_for_ref.append(torch.tensor(word_id).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(word_mask).unsqueeze(0))
                sentence_raw_for_re.append(sentence_raw)

            self.input_ids.append(sentences_for_ref)
            self.word_masks.append(attentions_for_ref)
            self.all_sentences.append(sentence_raw_for_re)

            llm_sentences_for_ref=[]
            llm_attentions_for_ref=[]
            llm_sentence_raw_for_re = []

            # add llm_output
            llm_sents = llm_out(os.path.join(self.llm_out_dir, str(r)+'.json'))
            if dataset=='refcocog' and splitBy == 'umd':
                # refcocog的llm文本是按照google的划分命名的
                google_r = np.load('{}/refer/refcocog/umd_2_google_ref.npy'.format(refer_data_root), allow_pickle=True).item()[r]
                llm_sents = llm_out(os.path.join(self.llm_out_dir, str(google_r)+'.json'))

            for i, llm_sent in enumerate(llm_sents):
                word_id = [self.tokenizer(sent).squeeze(0)[:self.max_tokens] for sent in llm_sent[:]]
                # if len(word_id) != 5:
                #     print(torch.stack(word_id).shape, r)
                word_id = torch.stack(word_id)
                # word_mask = torch.stack(word_id>0,dtype=int)
                
                llm_sentences_for_ref.append(torch.tensor(word_id).unsqueeze(0))
                # llm_attentions_for_ref.append(torch.tensor(word_mask).unsqueeze(0))
                llm_sentence_raw_for_re.append(llm_sent)
            
            self.llm_input_ids.append(llm_sentences_for_ref)
            # self.llm_word_masks.append(llm_attentions_for_ref)
            self.llm_all_sentences.append(llm_sentence_raw_for_re)
            
            semantic_cls = self.refer.Cats[ref['category_id']]
            semantic_prompt = 'a photo of {}.'.format(semantic_cls)
            self.semantic_coco_id.append(ref['category_id'])  # max: 90, min: 1
            self.semantic_word_ids.append(self.tokenizer(semantic_prompt).squeeze(0)[:self.max_tokens])
        
        print('Dataset prepared!')


        self.all_zs_masks = self.get_all_zs_masks()

    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self,index): # 36482
        this_ref_id=self.ref_ids[index]
        this_img_id=self.refer.getImgIds(this_ref_id)
        this_img=self.refer.Imgs[this_img_id[0]]
        
        img_path = os.path.join(self.refer.IMAGE_DIR,this_img['file_name'])
        img=Image.open(img_path).convert("RGB")

        ref=self.refer.loadRefs(this_ref_id)[0]
        
        # semantic_word_id
        # category_word_id = self.semantic_word_ids[index]
        # category_id = self.semantic_coco_id[index]
        
        ## box format: x1y1x2y2
        bbox = self.refer.Anns[ref['ann_id']]['bbox']
        bbox = np.array(bbox, dtype=int)
        bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]

        ref_mask=np.array(self.refer.getMask(ref)['mask'])
        annot=np.zeros(ref_mask.shape)
        annot[ref_mask==1]=1 
        # convert it to a Pillow image
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        pseudo_gt = None 
        
        # obtain sam_top_k mask
        ref_sam_out = self.sam_out[this_ref_id]
        # zs_mask_path = ref_sam_out['sam_mask_path']
        # zs_mask = torch.Tensor(np.load(zs_mask_path))  # 80 x H x W, faster training
        zs_mask = self.all_zs_masks[this_img['file_name'].replace('.jpg', '.npy')]
        zs_clip_socre = ref_sam_out['clip_score']
        candidate_id = ref_sam_out['candidate_2']
        

        if self.image_transforms is not None:
            h, w = ref_mask.shape 
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)
            # if not self.eval_mode:
            #     zs_mask = F.resize(zs_mask, (10, 10), interpolation=InterpolationMode.NEAREST)
            # else:
            #     sam_mask = F.resize(sam_mask, (10, 10), interpolation=InterpolationMode.NEAREST)
        else:
            target=annot
            zs_mask = zs_mask
        
        llm_word_ids = None
        llm_sents = None
        if self.eval_mode:
            embedding=[]
            att=[]
            sentences = [] 

            llm_embedding=[]
            llm_att=[]
            llm_sentences = []

            for s in range(len(self.input_ids[index])):
                e=self.input_ids[index][s]
                a=self.word_masks[index][s]
                sent = self.all_sentences[index][s]

                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
                sentences.append(sent)
                try:
                    llm_e=self.llm_input_ids[index][s]
                    # llm_a=self.llm_word_masks[index][s]
                    llm_sent = self.llm_all_sentences[index][s]

                    llm_embedding.append(llm_e.unsqueeze(-1))
                    # llm_att.append(llm_a.unsqueeze(-1))
                    llm_sentences.append(llm_sent)
                except:
                    breakpoint()

            # all sentence
            word_ids = torch.cat(embedding, dim=-1)
            word_masks = torch.cat(att, dim=-1)

            # llm
            llm_word_ids = torch.cat(llm_embedding, dim=-1)
            llm_sents = llm_sentences

            zs_mask = zs_mask
            zs_clip_socre = zs_clip_socre
        else: 
            # for training, random select one sentence 
            choice_sent = np.random.choice(len(self.input_ids[index]))
            word_ids = self.input_ids[index][choice_sent]
            # word_masks = self.word_masks[index][choice_sent]
            sentences = self.all_sentences[index][choice_sent]
            
            # llm
            llm_word_ids = self.llm_input_ids[index][choice_sent]
            llm_sents = self.llm_all_sentences[index][choice_sent]

            # SAM
            zs_mask = zs_mask
            zs_clip_socre = zs_clip_socre[choice_sent]
            candidate_id = candidate_id[choice_sent]
            if self.negative_samples > 0:
                neg_sents, neg_word_ids, diff_ref_id, diff_ref_choice_sent, same_img_mark_0  = self.get_neg_samples(this_img_id, this_ref_id, sentences)
                diff_ref_word_ids_0 = neg_word_ids[-1]
                diff_ref_index = self.ref_ids.index(diff_ref_id)
                try:
                    diff_ref_llm_word_ids_0 = self.llm_input_ids[diff_ref_index][diff_ref_choice_sent]
                except:
                    breakpoint()
                    
                # neg_sents, neg_word_ids, diff_ref_id, diff_ref_choice_sent, same_img_mark_1  = self.get_neg_samples(this_img_id, this_ref_id, sentences)
                # diff_ref_word_ids_1 = neg_word_ids[-1]
                # diff_ref_index = self.ref_ids.index(diff_ref_id)
                # try:
                #     diff_ref_llm_word_ids_1 = self.llm_input_ids[diff_ref_index][diff_ref_choice_sent]
                # except:
                #     breakpoint()

        img_path_full = this_img['file_name']
        img_path = int(img_path_full.split('.')[0].split('_')[-1])
        # print(zs_mask.shape, zs_clip_socre.shape, candidate_id.shape)
        samples = {
            "ref_id": this_ref_id,
            "img": img,
            "word_ids": word_ids,
            # "word_masks": word_masks,
            "target": target.unsqueeze(0),
            "category": self.refer.Cats[ref['category_id']],
            # "category_id": category_id,
            # "category_word_id": category_word_id,
            'zs_mask':zs_mask,
            'clip_score': zs_clip_socre,
            'candidate_id':candidate_id
        }
        if self.negative_samples > 0:
            samples['neg_sents'] = neg_sents
            samples['neg_word_ids'] = neg_word_ids

            samples['diff_ref_word_ids_0'] = diff_ref_word_ids_0
            samples['diff_ref_llm_word_ids_0'] = diff_ref_llm_word_ids_0
            samples['same_img_mark_0'] = same_img_mark_0
            
            # samples['diff_ref_word_ids_1'] = diff_ref_word_ids_1
            # samples['diff_ref_llm_word_ids_1'] = diff_ref_llm_word_ids_1
            # samples['same_img_mark_1'] = same_img_mark_1
            
        targets = {
            "target": target.unsqueeze(0),
            "img_path": img_path,
            "sentences": sentences,
            "boxes": bbox,
            "orig_size": np.array([h, w]),
            "img_path_full": img_path_full
        }
     
        if pseudo_gt is not None:
            targets['pseudo_gt'] = pseudo_gt

        if (llm_word_ids is not None) and (llm_sents is not None):
            samples['llm_word_ids'] =  llm_word_ids
            samples['llm_sents'] = llm_sents
        # print(llm_word_ids.shape)
        # print(candidate_id.shape, zs_clip_socre.shape, zs_mask.shape)
        return samples, targets

    def get_neg_samples(self, this_img_id, this_ref_id, sentences):
        ###########
        img2ref = self.refer.imgToRefs[this_img_id[0]]
        neg_index = []
        neg_ref_ids = []
        for item in img2ref:
            t_ref_id = item['ref_id']
            t_category_id = item['category_id']
            try:
                if t_ref_id != this_ref_id:  # and this_category_id == t_category_id
                    neg_index.append(self.refid2index[t_ref_id])
                    neg_ref_ids.append(t_ref_id)
            except: ### for refcocog google, its refindex is not match
                break 
                import pdb
                pdb.set_trace() 
        ###########

        if len(neg_index) > 0:
            neg_sents = []
            neg_word_ids = []
            ## random select negtive samples from same random index 
            # n_index = neg_index[np.random.choice(len(neg_index))]
            while len(neg_sents) < self.negative_samples:
                ## different random index 
                choice_ref = np.random.choice(len(neg_index))
                n_index = neg_index[choice_ref]
                choice_sent = np.random.choice(len(self.input_ids[n_index]))
                neg_word_ids.append(self.input_ids[n_index][choice_sent])
                neg_sents.append(self.all_sentences[n_index][choice_sent])
            neg_word_ids = torch.cat(neg_word_ids, dim=0)
            
            return neg_sents, neg_word_ids, neg_ref_ids[choice_ref], choice_sent, True
        else:
            # random index, then randomly select one sentence 
            neg_sents = []
            neg_word_ids = []
            while len(neg_sents) < self.negative_samples:
                n_index = np.random.choice(len(self.input_ids))
                choice_sent = np.random.choice(len(self.input_ids[n_index]))
                tmp_sent = self.all_sentences[n_index][choice_sent]
                if tmp_sent != sentences:
                    neg_sents.append(tmp_sent)
                    neg_word_ids.append(self.input_ids[n_index][choice_sent])
            neg_word_ids = torch.cat(neg_word_ids, dim=0)

            return neg_sents, neg_word_ids, self.ref_ids[n_index], choice_sent, False
    
    def get_all_zs_masks(self):
        
        all_zs_masks = {}
        import tqdm
        # zs_out_dir = zs_sam_out_dir
        # zs_out_dir = zs_solo_out_dir
        zs_out_dir = self.zs_out_dir
        zs_files = []
        
        print('loading zs_mask from {}.\n'.format(zs_out_dir) * 4)
        for ref_id in tqdm.tqdm(self.ref_ids):
            ref_sam_out = self.sam_out[ref_id]
            zs_mask_path = ref_sam_out['sam_mask_path']
            npy_name = zs_mask_path.split('/')[-1]
            zs_files.append(npy_name)
        zs_files = list(set(zs_files))

        for zs_file in tqdm.tqdm(zs_files):
            zs_mask = torch.Tensor(np.load(os.path.join(zs_out_dir, zs_file)))  # 80 x H x W
            if not self.eval_mode:
                zs_mask = F.resize(zs_mask.unsqueeze(dim=0), (32, 32), interpolation=InterpolationMode.NEAREST)[0]
            # all_zs_masks[ref_id] = zs_mask
            if zs_mask.shape[0] != 40:
                print(os.path.join(zs_out_dir, zs_file))
            if zs_file not in all_zs_masks.keys():
                all_zs_masks[zs_file] = zs_mask

        return all_zs_masks
                
if __name__ == '__main__':
    from transform import get_transform
    import numpy as np 
    import json 
    from torch.utils.data import DataLoader

    refcoco_train = ReferDataset(dataset='refcoco', splitBy='unc', split='train', eval_mode=False, image_transforms=get_transform(320, train=False)) 
    train_loader=DataLoader(refcoco_train,
                            batch_size=12,
                            num_workers=2,
                            pin_memory=True,
                            sampler=None)
    for idx,(img, target, bbox, word_ids, word_mask, _, raw_sentences) in enumerate(train_loader):
        print(idx, img.shape)

        if idx > 10: break 