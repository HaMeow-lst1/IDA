# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import numpy as np
import torch.nn as nn
import IDA_config

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


class HiNet(nn.Module):
    
    
    
    
    def __init__(self, k = 32):
        super(HiNet, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(stride = 4, kernel_size = 4))
            
        self.dense_mean_33 = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
        
        self.dense_var_33 = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
        
        
        self.dense_mean_34 = nn.Sequential(
            nn.Linear(256 * 3 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
        
        self.dense_var_34 = nn.Sequential(
            nn.Linear(256 * 3 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
        
        self.dense_mean_45 = nn.Sequential(
            nn.Linear(256 * 4 * 5, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
        
        self.dense_var_45 = nn.Sequential(
            nn.Linear(256 * 4 * 5, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
            
        self.dense_mean_56 = nn.Sequential(
            nn.Linear(256 * 5 * 6, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
        
        self.dense_var_56 = nn.Sequential(
            nn.Linear(256 * 5 * 6, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, self.k))
    
    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        if x.shape[-1] == 3:
            x = x.view(-1, 256 * 3 * 3)
            return self.dense_mean_33(x), self.dense_var_33(x)
        
        elif x.shape[-1] == 4:
            x = x.view(-1, 256 * 3 * 4)
            return self.dense_mean_34(x), self.dense_var_34(x)
        
        elif x.shape[-1] == 5:
            x = x.view(-1, 256 * 4 * 5)
            return self.dense_mean_45(x), self.dense_var_45(x)
        else:
            x = x.view(-1, 256 * 5 * 6)
            return self.dense_mean_56(x), self.dense_var_56(x)


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """
    
    
    ida = IDA_config.IDA

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        
        if self.ida.is_ida:
            self.k = self.ida.K
            self.hinet = HiNet(k = self.k)
        
            self.lambda_mean = self.ida.lambda_mean
            self.lambda_var = self.ida.lambda_mean

    def extract_feat(self, img, is_ida = False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if is_ida:
            prediction_mean, prediction_var = self.hinet(x[-1])
        if self.with_neck:
            x = self.neck(x)
        if is_ida:
            return x, prediction_mean, prediction_var
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        if self.ida.is_ida:
            x, prediction_mean, prediction_var = self.extract_feat(img, is_ida = True)
        else:
            x = self.extract_feat(img, is_ida = False)
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self.ida.is_ida:
                                              
            gt_mean_lst, gt_var_lst = [], []
        
            for item in img_metas:
                gt_mean_lst.append(item['hi_ann'][0].reshape((1, -1)))
                gt_var_lst.append(item['hi_ann'][1].reshape((1, -1)))
        
            gt_mean = np.concatenate(gt_mean_lst, axis = 0)
            gt_var = np.concatenate(gt_var_lst, axis = 0)    
        
        
            gt_mean = torch.tensor(gt_mean).cuda()
            gt_var = torch.tensor(gt_var).cuda()
        
            bat = len(img_metas)
            #print(gt_mean.shape)
            #print(prediction_mean.shape)
            loss_mean = self.lambda_mean * bat * torch.mean(torch.pow((gt_mean - prediction_mean), 2)).float()
            loss_var = self.lambda_var * bat * torch.mean(torch.pow((gt_var - prediction_var), 2)).float()
        
        
            losses['loss_ida_mean'] = loss_mean
            losses['loss_ida_var'] = loss_var
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
