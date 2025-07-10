import sys
sys.path.append("/home/yzq/mnt/RIS/segmentation/TRIS_2024")

from showsolo.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import cv2 as cv

dataset_dir = '/home/yzq/mnt/RIS/coco/train2014/'
json_file = '/home/yzq/mnt/RIS/coco/annotations/instances_train2014.json'
coco = COCO(json_file)
# 展示coco的所有类别
print(coco.cats)
# catIds = coco.getCatIds(catNms=['cat', 'person']) # catIds=1,2 表示同时含有人和猫这一类
catIds = coco.getCatIds(catNms=['giraffe']) # catIds=1 表示人这一类
# catIds = coco.getCatIds(catNms=['yuanhuan', 'yellow', 'green', 'red', 'indicator']) # catIds=1 表示人这一类
imgIds = coco.getImgIds(catIds=catIds ) # 图片id，许多值
img_num = 0
for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    img_num += 1
    print(img['file_name'])
    if img['file_name'] == "COCO_train2014_000000152253.jpg":
        I = io.imread(dataset_dir + img['file_name'])
    
        plt.axis('off')
        plt.imshow(I) #绘制图像，显示交给plt.show()处理
        # annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        # 加了 catIds 就是只加载目标类别的anno，不加就是图像中所有的类别anno
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        plt.savefig('./ans.png')
        break
    else:
        print('there is no this image.')
print(img_num)

