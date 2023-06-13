# IDA for others.

## Dataset

Please download the dataset here:<a href="https://drive.google.com/drive/folders/1FiP1S0VdBVgQbyYWy7U8Tf1LsWXI4K1c?usp=sharing">datasets </a>. All these are VOC2007 format.

## Pre-training models

Please download the pre-training models here: <a href="https://drive.google.com/drive/folders/1PBcThV9qMB9ZXi8X9lwNAScuAp4sBVds?usp=sharing">pre_models </a>.

## Training

### Step 0 (Before training)

Run

```
cd MMdetection
pip install -v -e .
```

Please open the IDA_config.py, change the function "get_dataset_path" according to the path of datasets. Change the "IDA.pre_model_path" according to the path of pre_training models.

### Step 1

Please change IDA_config.py as follows:

```
IDA.is_ida = False
IDA.mode = 'thermal'
```

Besides, please ensure "IDA.dataset" and "IDA.model". Then choose one to run 

```
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_kaist_thermal.py
python tools/train.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_kaist_thermal.py
python tools/train.py configs/pascal_voc/yolov3_thermal.py
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_flir_thermal.py
python tools/train.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_flir_thermal.py
```

After this step, models are saved under "work_dirs" folder.

### Step 2

Please ensure the "IDA.thermal_model_path" and "IDA.K" from IDA_config.py , and run

```
python collect_feature_thermal.py
```

Note that "IDA.thermal_model_path" should be the path of model trained in the Step 1. After this step, "thermal_feature.npy" is saved.

### Step 3

Please change IDA_config.py as follows:

```
IDA.is_ida = False
IDA.mode = 'visible'
```

Besides, please ensure "IDA.dataset" and "IDA.model". Then choose one to run 

```
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_kaist_visible.py
python tools/train.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_kaist_visible.py
python tools/train.py configs/pascal_voc/yolov3_visible.py
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_flir_visible.py
python tools/train.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_flir_visible.py
```

After this step, models are saved under "work_dirs" folder.

### Step 4

Please ensure the "IDA.visible_model_path" and "IDA.K" from IDA_config.py , and run

```
python collect_feature_visible.py
```

Note that "IDA.visible_model_path" should be the path of model trained in the Step 3. After this step, "visible_feature.npy" is saved.

### Step 5

Please ensure the "IDA.K", "IDA.N", "IDA.lambda_mean", and "IDA.lambda_var" from IDA_config.py , and run

```
python feco.py
```

After this step, "meanvar_dic.npy" is saved.

### Step 6

Please change IDA_config.py as follows:

```
IDA.is_ida = True
IDA.mode = 'thermal'
```

Besides, please ensure "IDA.dataset" and "IDA.model". Then choose one to run

```
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_kaist.py
python tools/train.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_kaist.py
python tools/train.py configs/pascal_voc/yolov3.py
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_flir.py
python tools/train.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_flir.py
```

After this step, models are saved under "work_dirs" folder.

## Evaluation

Please run

```
python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_kaist.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712_kaist/latest.pth --out ./detection_results.pkl
python tools/test.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_kaist.py work_dirs/cascade_rcnn_r50_fpn_voc0712_kaist/latest.pth --out ./detection_results.pkl
python tools/test.py configs/pascal_voc/yolov3.py work_dirs/yolov3/latest.pth --out ./detection_results.pkl
python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_flir.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712_flir/latest.pth --out ./detection_results.pkl
python tools/test.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_flir.py work_dirs/cascade_rcnn_r50_fpn_voc0712_flir/latest.pth --out ./detection_results.pkl
```

For the KAIST dataset, please run

```
python lamr_ap.py
```

For the FLIR-aligned dataset, please run

```
python tools/analysis_tools/eval_metric.py configs/pascal_voc/yolov3.py ./detection_results.pkl --eval mAP
python tools/analysis_tools/eval_metric.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_flir.py ./detection_results.pkl --eval mAP
python tools/analysis_tools/eval_metric.py configs/pascal_voc/cascade_rcnn_r50_fpn_voc0712_flir.py ./detection_results.pkl --eval mAP
```

Also, you can download our finished models here: <a href="https://drive.google.com/drive/folders/1cfRVg-pfxIwmjAzvwkg6WVpwo47ECUwu?usp=sharing">models </a>.

## Citation

```latex
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}

```
