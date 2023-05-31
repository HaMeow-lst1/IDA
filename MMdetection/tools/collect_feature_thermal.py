import asyncio
import numpy as np
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

import mmcv
import os

import warnings
warnings.filterwarnings("ignore") 


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        
        return imagelist

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference') #cuda:0
    #parser.add_argument('--out_dir', default='./output/',help='Image file')
    parser.add_argument('--config',default='configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py', help='Config file')
    parser.add_argument('--checkpoint',default='work_dirs/faster_rcnn_r50_fpn_1x_voc0712/latest.pth', help='Checkpoint file')
    parser.add_argument('--dataset_path',  default='/home/lisongtao/datasets/KaistThermalVoc/', help='bbox score threshold')

    args = parser.parse_args()
    return args


def main(args):
    
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    img = mmcv.imread('/home/lisongtao/datasets/KaistThermalVoc/VOC2007/JPEGImages/set00_V000_lwir_I01225.png')
    
    print(img)
    print(type(img))
    
    
    '''
    
    for image in images:
      print(image)
	 # 测试单张图片并展示结果
      img = mmcv.imread(image)    # 或者 ，这样图片仅会被读一次 img = 'demo.jpg'
      result = inference_detector(model, img)

      out_file = out_dir + image.split('/')[-1]
      model.show_result(img,result, score_thr=args.score_thr,out_file=out_file)
    '''
    




if __name__ == '__main__':
    args = parse_args()
    main(args)