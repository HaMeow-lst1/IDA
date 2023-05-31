import asyncio
import numpy as np
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot, get_feature)

import mmcv
import os

import torch

from tqdm import tqdm
import IDA_config

import warnings
warnings.filterwarnings("ignore") 





def main():
    ida = IDA_config.IDA

    device = 'cuda:0'

    feature_dic = {}
    
    k = ida.K

    f_min = np.full(k, +10000.0)
    f_max = np.full(k, -10000.0)
    
    model = init_detector(ida.model_config, ida.visible_model_path, device=device)
    
    with open(ida.dataset_path + 'VOC2007/ImageSets/Main/trainval.txt', 'r') as f:
        names = f.readlines()
        f.close()
    for name in tqdm(names):
        
        if ida.dataset == 'KAIST':
            img = mmcv.imread(ida.dataset_path + 'VOC2007/JPEGImages/' + name[:-1] + '.png')
        if ida.dataset == 'FLIR':
            img = mmcv.imread(ida.dataset_path + 'VOC2007/JPEGImages/' + name[:-1].replace('PreviewData', 'RGB') + '.jpg')
    
        feature = get_feature(model, img)
        feature = torch.mean(feature, dim = (2, 3))
        feature = feature.view(k, -1)
        feature = torch.mean(feature, dim = 1)
        feature = feature.cpu().detach().numpy()
        feature_dic[name[:-1] + '.png'] = feature
        for i in range(k):
            if feature[i] > f_max[i]:
                f_max[i] = feature[i]
            if feature[i] < f_min[i]:
                f_min[i] = feature[i]
    for key in feature_dic.keys():
        feature_dic[key] = (feature_dic[key] - f_min) / (f_max - f_min)
    
    np.save('visible_feature.npy', feature_dic)
    print('finish save')
    
    

if __name__ == '__main__':
    main()