# IDA for YOLOv3 on the KAIST dataset.

## Dataset

Please download the dataset here:<a href="https://drive.google.com/file/d/14A3K2IPPPC8-BwPh-YjeHARaZqjnR655/view?usp=sharing">KAIST_dataset </a>. Then modify the contents of the four files under the "data" folder: train_thermal.txt, train_visible.txt, test_thermal.txt, and test_visible.txt. Please change the path of the dataset.

## Pre-training models

Please download the pre-training models here: <a href="https://drive.google.com/file/d/1Kyoyira0liRRr_FOY8DDSeATLQAwXtu-/view?usp=sharing">kaist_thermal_detector.weights </a> and <a href="https://drive.google.com/file/d/1xiSKTNEB5ng0T5kgyjUKytlpn3q84uK6/view?usp=sharing">kaist_visible_detector.weights </a>. Please put them under "weights" folder.

##Training

Run

```
cd TaskConditioned
```

Step 1: Please change IDA_config.py as follows:

```
IDA.is_ida = False
IDA.mode = 'thermal'
```

Then run 

```
python train.py
```

After this step, models are saved under "backupthermal" folder.

Step 2: Please change IDA_config.py as follows:

```
IDA.is_ida = False
IDA.mode = 'visible'
```

Then run 

```
python train.py
```

After this step, models are saved under "backupvisible" folder.
