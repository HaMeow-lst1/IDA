from easydict import EasyDict as edict

def get_dataset_path(mode, dataset):    #path of dataset
    if dataset == 'KAIST':
        if mode == 'thermal':
            return '/home/lisongtao/datasets/KaistThermalVoc/'
        if mode == 'visible':
            return '/home/lisongtao/datasets/KaistVisibleVoc/'
    if dataset == 'FLIR':
        return '/home/lisongtao/datasets/FLIRVOC/'

def get_dataset_times(dataset):
    if dataset == 'KAIST':
        return 1
    if dataset == 'FLIR':
        return 2

def get_model_config(model, dataset, mode):
    if model == 'YOLOv3':
        if mode == 'thermal':
            return 'configs/pascal_voc/yolov3_thermal.py'
        if mode == 'visible':
            return 'configs/pascal_voc/yolov3_visible.py'
    if model == 'FasterRCNN':
        if dataset == 'KAIST':
            if mode == 'thermal':
                return 'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_kaist_thermal.py'
            if mode == 'visible':
                return 'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_kaist_visible.py'
        if dataset == 'FLIR':
            if mode == 'thermal':
                return 'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_flir_thermal.py'
            if mode == 'visible':
                return 'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_flir_visible.py'
    if model == 'CascadeRCNN':
        if dataset == 'KAIST':
            if mode == 'thermal':
                return 'configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_kaist_thermal.py'
            if mode == 'visible':
                return 'configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_kaist_visible.py'
        if dataset == 'FLIR':
            if mode == 'thermal':
                return 'configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_flir_thermal.py'
            if mode == 'visible':
                return 'configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_flir_visible.py'

        


IDA = edict()

IDA.is_ida = True
IDA.mode = 'thermal'    #thermal or visible, is_ida = True && mode = 'thermal'
IDA.dataset = 'KAIST'   #['KAIST', 'FLIR']
IDA.model = 'CascadeRCNN'  #['YOLOv3', 'FasterRCNN', 'CascadeRCNN']

IDA.pre_model_path = {'YOLOv3': '/home/lisongtao/models/mmdetection/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth', 
                      'FasterRCNN': '/home/lisongtao/models/mmdetection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', 
                      'CascadeRCNN': '/home/lisongtao/models/mmdetection/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'}





IDA.dataset_path = get_dataset_path(IDA.mode, IDA.dataset)
IDA.dataset_times = get_dataset_times(IDA.dataset)
IDA.model_config = get_model_config(IDA.model, IDA.dataset, IDA.mode)


#  (K, N, lambda_mean, lambda_var) in paper
##########################################################################################################
#          #                       #                         #                                                                                                                                                                
#          #        YOLOv3         #         FasterRCNN      #        CascadeRCNN                                                               
#          #                       #                         #                                                       
##########################################################################################################
#          #                       #                         #                                                      
#   KAIST  #           x           #  (32, 40, 0.1, 0.125)   #       (32, 40, 0.1, 0.2)                                                    
#          #                       #                         #                                                      
##########################################################################################################
#          #                       #                         #                                                      
#   FLIR   # (32, 40, 100.0, 200.0)#  (32, 40, 0.1, 0.125)   #       (16, 40, 0.1, 0.2)                                                   
#          #                       #                         #                                                      
##########################################################################################################


IDA.K = 32
IDA.N = 40
IDA.lambda_mean = 0.1
IDA.lambda_var = 0.2

'''
work_dirs/faster_rcnn_r50_fpn_1x_voc0712_kaist_thermal/latest.pth
work_dirs/cascade_rcnn_r50_fpn_voc0712_kaist_thermal/latest.pth
work_dirs/yolov3_thermal/latest.pth
work_dirs/faster_rcnn_r50_fpn_1x_voc0712_flir_thermal/latest.pth
work_dirs/cascade_rcnn_r50_fpn_voc0712_flir_thermal/latest.pth
'''
IDA.thermal_model_path = 'work_dirs/faster_rcnn_r50_fpn_1x_voc0712_kaist_thermal/latest.pth'  

'''
work_dirs/faster_rcnn_r50_fpn_1x_voc0712_kaist_visible/latest.pth
work_dirs/cascade_rcnn_r50_fpn_voc0712_kaist_visible/latest.pth
work_dirs/yolov3_visible/latest.pth
work_dirs/faster_rcnn_r50_fpn_1x_voc0712_flir_visible/latest.pth
work_dirs/cascade_rcnn_r50_fpn_voc0712_flir_visible/latest.pth
'''
IDA.visible_model_path = 'work_dirs/faster_rcnn_r50_fpn_1x_voc0712_kaist_visible/latest.pth'  

'''
work_dirs/faster_rcnn_r50_fpn_1x_voc0712_kaist/latest.pth
work_dirs/cascade_rcnn_r50_fpn_voc0712_kaist/latest.pth
work_dirs/yolov3/latest.pth
work_dirs/faster_rcnn_r50_fpn_1x_voc0712_flir/latest.pth
work_dirs/cascade_rcnn_r50_fpn_voc0712_flir/latest.pth
'''
#IDA.model_path = 'work_dirs/yolov3/latest.pth' 