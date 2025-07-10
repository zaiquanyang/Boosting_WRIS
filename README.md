## Boosting_WRIS
--- 
Official code of our paper ["Weakly-Supervised Referring Image Segmentation via Progressive Comprehension"](https://arxiv.org/pdf/2410.01544), NeurIPS2024

### Overview
---
![aa](./QQ20250710-213401.png)

we propose a  Progressive Comprehension Network (PCNet) to leverage target-related textual cues from the input description for progressively localizing the target object. We first use a Large Language Model (LLM) to decompose the input text description into short phrases. These short phrases are taken as target-related cues and fed into a Conditional Referring Module (CRM) in multiple stages, to enhance the response map for target localization in a multi-stage manner. We also propose a RaS loss to constrain the visual localization across different stages and an Instance-aware Disambiguation (IaD) loss to suppress instance localization ambiguity.


### Preparation

- Data
  - extract phrase by LLM (Mistral or others LLMs)
  - extract mask proposals by SAM

    The overall dataset structure is as follow:

    ```
    coco/
    ├── annotations/
    │   ├── captions_train2014.json
    │   ├── instances_train2014.json
    ├── refer/
    │   ├── grefcoco/
    │   ├── refcoco/
    │   ├── refcoco+/
    │   │   ├── Mistral_7B_Ref_Sents/
    │   │   ├── SAM_Mask/
    │   │   ├── SOLO_Mask/
    │   │   ├── instances.json
    │   │   └── refs(unc).p
    │   ├── refcocog/
    │   │   └── new_grefs_unc.json
    │   ├── refcoco.zip
    │   ├── refcoco+.zip
    │   └── refcocog.zip
    ├── train2014/
        ├── COCO_train2014_000000000009.jpg
        ├── COCO_train2014_000000000025.jpg
        ├── COCO_train2014_000000000034.jpg
    ```

    Replace the `refer_data_root in line94 of dataset/ReferDataset` with the path of `coco`


- Environment

  ```
  conda env create -f environment.yml

### Evaluation

- Download Checkpoint [weights](TODO)
  
```shell
dataset=refcoco+
splitBy=unc
test_split=val
model_path="weights/ckpt_hit.pth"
out_put_dir=""


CUDA_VISIBLE_DEVICES=0 python validate.py \
    --batch_size 1 \
    --size 320 \
    --dataset ${dataset} \
    --splitBy ${splitBy} \
    --test_split ${test_split} \
    --max_query_len 20 \
    --attn_multi_vis 0.1 \
    --attn_multi_text 0.1 \
    --output ${out_put_dir} \
    --resume \
    --K_Iters 1 \
    --pretrain  "${model_path}" \
    --cam_save_dir "${out_put_dir}/cam_${test_split}" \
    --name_save_dir "${out_put_dir}/name_save" \
    --save_train_pseudo_dir "${out_put_dir}/train_pseudo" \
    --save_cam \
    --eval
```

### Acknowledgement

This repository  was mostly based on [TRIS](https://github.com/fawnliu/TRIS/tree/main)

### Citation

```
```

### Contact
If you have any questions, please feel free to reach out at zaiquanyangcat@gmail.com.