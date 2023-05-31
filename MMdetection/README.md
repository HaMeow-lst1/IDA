# IDA for others.

## Dataset

Please download the dataset here:<a href="https://drive.google.com/file/d/14A3K2IPPPC8-BwPh-YjeHARaZqjnR655/view?usp=sharing">KAIST_dataset </a>. Then modify the contents of the four files under the "data" folder: train_thermal.txt, train_visible.txt, test_thermal.txt, and test_visible.txt. Please change the path of the dataset.

## Pre-training models

Please download the pre-training models here: <a href="https://drive.google.com/drive/folders/1PBcThV9qMB9ZXi8X9lwNAScuAp4sBVds?usp=sharing">pre_models </a>.

## Training

Run

```
cd TaskConditioned
```

### Step 1

Please change IDA_config.py as follows:

```
IDA.is_ida = False
IDA.mode = 'thermal'
```

Then run 

```
python train.py
```

After this step, models are saved under "backupthermal" folder.

### Step 2

Please ensure the "IDA.thermal_model_path" and "IDA.K" from IDA_config.py , and run

```
python collect_feature_thermal.py
```

Note that "IDA.thermal_model_path" should be the path (".weights" not ".model") of model trained in the Step 1. After this step, "thermal_feature.npy" is saved.

### Step 3

Please change IDA_config.py as follows:

```
IDA.is_ida = False
IDA.mode = 'visible'
```

Then run 

```
python train.py
```

After this step, models are saved under "backupvisible" folder.

### Step 4

Please ensure the "IDA.visible_model_path" and "IDA.K" from IDA_config.py , and run

```
python collect_feature_visible.py
```

Note that "IDA.visible_model_path" should be the path (".weights" not ".model") of model trained in the Step 3. After this step, "visible_feature.npy" is saved.

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

Then run

```
CUDA_VISIBLE_DEVICES = 0 python train.py
```
(TODO: multi-gpu for training in this step.)

After this step, models are saved under "backup" folder.

## Evaluation

Please ensure the "IDA.model_path" from IDA_config.py. Note that "IDA.model_path" should be the finished training model. Then run

```
python evaluation.py
```

Also, you can download our finished models here: <a href="https://drive.google.com/drive/folders/14o9pdR3L6eRIVNe8wi-V4SqOn7uQoara?usp=share_link">models </a>.
