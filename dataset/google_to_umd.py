import pickle
import numpy as np
import os
import shutil
# google_refs = pickle.load(open('/home/yzq/mnt/RIS/coco/refer/refcocog/refs(google).p', 'rb'))
# umd_refs = pickle.load(open('/home/yzq/mnt/RIS/coco/refer/refcocog/refs(umd).p', 'rb'))

# umd_ref_dict = {}

# for umd_ref in umd_refs:
#     umd_ref_id = umd_ref['ref_id']
#     umd_sent_id = umd_ref['sent_ids']

#     for google_ref in google_refs:
#         if umd_sent_id == google_ref['sent_ids']:
#             umd_ref_dict[umd_ref_id] = google_ref['ref_id']
#             print(umd_ref_id, google_ref['ref_id'])
        

# np.save('/home/yzq/mnt/RIS/coco/refer/refcocog/umd_2_google_ref.npy', umd_ref_dict)

zs_model = "SOLO_Mask"
refcoco_train = os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco', zs_model, '{}_{}_mIoU_27.7_oIoU_24.6_candidate_2_4_6_8.npy'.format('unc', 'train'))
refcoco_testA = os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco', zs_model, '{}_{}_mIoU_26.9_oIoU_24.1_candidate_2_4_6_8.npy'.format('unc', 'testA'))
refcoco_testB = os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco', zs_model, '{}_{}_mIoU_26.7_oIoU_24.3_candidate_2_4_6_8.npy'.format('unc', 'testB'))
refcoco_val = os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco', zs_model, '{}_{}_mIoU_27.5_oIoU_24.7_candidate_2_4_6_8.npy'.format('unc', 'val'))

refcoco_train_dict = np.load(refcoco_train, allow_pickle=True).item()
refcoco_testA_dict = np.load(refcoco_testA, allow_pickle=True).item()
refcoco_testB_dict = np.load(refcoco_testB, allow_pickle=True).item()
refcoco_val_dict = np.load(refcoco_val, allow_pickle=True).item()

# print(len(refcocg_google_val_dict.keys()), len(refcocg_google_train_dict.keys()))

all_dict = {}

for ref_id in refcoco_train_dict.keys():
    # assert ref_id not in refcocg_google_train_dict.keys()
    all_dict[ref_id] = refcoco_train_dict[ref_id]

for ref_id in refcoco_testA_dict.keys():
    # assert ref_id not in refcocg_google_train_dict.keys()
    all_dict[ref_id] = refcoco_testA_dict[ref_id]

for ref_id in refcoco_testB_dict.keys():
    # assert ref_id not in refcocg_google_train_dict.keys()
    all_dict[ref_id] = refcoco_testB_dict[ref_id]

for ref_id in refcoco_val_dict.keys():
    # assert ref_id not in refcocg_google_train_dict.keys()
    all_dict[ref_id] = refcoco_val_dict[ref_id]

# refcocg_google = refcocg_google_train_dict
print(len(all_dict.keys()))

SAM_dict_file = os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco/SAM_Mask', 'unc_candidate_2_4_6_8.npy')

SAM_dict = np.load(SAM_dict_file, allow_pickle=True).item()

for key in SAM_dict.keys():
    if key not in list(all_dict.keys()):
        all_dict[key] = SAM_dict[key]
        src = os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco/SAM_Mask/SAM_Out', SAM_dict[key]['sam_mask_path'])
        dst = os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco/', 'SOLO_Mask', 'SOLO_out', SAM_dict[key]['sam_mask_path'])
        assert os.path.exists(src), print('SAM file {} not exist'.format(SAM_dict[key]['sam_mask_path']))
        # breakpoint()
        shutil.copyfile(src, dst)
        assert os.path.exists(dst)
breakpoint()
np.save(os.path.join('/home/yzq/mnt/RIS/coco/refer/refcoco', zs_model, 'unc_candidate_2_4_6_8.npy'), all_dict)

# refcocog_google_dict = np.load('/home/yzq/mnt/RIS/coco/refer/refcocog/SAM_Mask/google_candidate_2_4_6_8.npy', allow_pickle=True).item()
# umd_2_google_ref = np.load('/home/yzq/mnt/RIS/coco/refer/refcocog/umd_2_google_ref.npy', allow_pickle=True).item()

# refcocog_umd_dict = {}

# for ref_id in umd_2_google_ref.keys():
#     refcocog_umd_dict[ref_id] = refcocog_google_dict[umd_2_google_ref[ref_id]]

# print(len(refcocog_umd_dict.keys()))
# np.save(os.path.join('/home/yzq/mnt/RIS/coco/refer/refcocog', zs_model, 'umd_candidate_2_4_6_8.npy'), refcocog_umd_dict)
