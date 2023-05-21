from easydict import EasyDict as edict


IDA = edict()

IDA.is_ida = False      
IDA.mode = 'thermal'    #thermal or visible, is_ida = True && mode = 'thermal'

IDA.K = 32
IDA.N = 40
IDA.lambda_mean = 0.5
IDA.lambda_var = 1.0

IDA.thermal_model_path = 'backupthermal/yolov3_kaist_000050.weights'  #.weights
IDA.visible_model_path = 'backupvisible/yolov3_kaist_000050.weights'  #.weights

IDA.model_path = 'backup/yolov3_kaist_000050.model' # the path of the finished traning model, .model