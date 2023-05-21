from __future__ import print_function
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms
import gc

import dataset
from utils import *
from image import correct_yolo_boxes
from cfg import parse_cfg
from darknet import Darknet
import argparse
import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import IDA_config

def main():
    ida = IDA_config.IDA
    k = ida.K
    feature_dic = {}
    
    
    
    datacfg    = FLAGS.data
    cfgfile    = FLAGS.config
    weightfile = weightfile = ida.visible_model_path
    eval    = FLAGS.eval
    continuetrain = FLAGS.continuetrain
    adaptation = FLAGS.adaptation
    layerwise = FLAGS.layerwise
    max_epochs = FLAGS.epoch
    # condition = FLAGS.condition

    data_options  = read_data_cfg(datacfg)
    net_options   = parse_cfg(cfgfile)[0]

    global use_cuda
    use_cuda = True
    globals()["trainlist"]     = data_options['train']
    globals()["testlist"]      = data_options['valid']
    globals()["classname"]     = data_options['names']
    globals()["backupdir"]     = data_options['backup']
    globals()["gpus"] = data_options['gpus']  # e.g. 0,1,2,3
    globals()["ngpus"]         = len(gpus.split(','))
    globals()["num_workers"]   = int(data_options['num_workers'])
    globals()["batch_size"]    = int(net_options['batch'])
    globals()["max_batches"]   = int(net_options['max_batches'])
    globals()["burn_in"]       = int(net_options['burn_in'])
    # globals()["learning_rate"] = float(net_options['learning_rate'])
    globals()["momentum"]      = float(net_options['momentum'])
    globals()["decay"]         = float(net_options['decay'])
    globals()["steps"]         = [int(step) for step in net_options['steps'].split(',')]
    globals()["scales"]        = [float(scale) for scale in net_options['scales'].split(',')]

    learning_rate = float(net_options['learning_rate'])
    try:
        globals()["backupdir"] = data_options['backup']
    except:
        globals()["backupdir"] = 'backup'

    if not os.path.exists(backupdir):
        os.mkdir(backupdir)

    try:
        globals()["logfile"] = data_options['logfile']
    except:
        globals()["logfile"] = 'backup/logfile.txt'

    try:
        globals()["condition"] = bool(net_options['condition'])
    except:
        globals()["condition"] = False

    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    global model
    model = Darknet(cfgfile, use_cuda=use_cuda)

    model.print_network()
    nsamples = file_lines(trainlist)
    #initialize the model
    if FLAGS.reset:
        model.seen = 0
        init_epoch = 0
    else:
        init_epoch = model.seen//nsamples
    iterates = 0

    


    if weightfile is not None:
        model.load_weights(weightfile)


    if continuetrain is not None:
        checkpoint = torch.load(continuetrain)
        model.load_state_dict(checkpoint['state_dict'])
        try:
            init_epoch = int(continuetrain.split('.')[0][-2:])
        except:
            logging('Warning!!! Continuetrain file must has at least 2 number at the end indicating last epoch')
        iterates = init_epoch*(nsamples/batch_size)



    global loss_layers
    loss_layers = model.loss_layers
    for l in loss_layers:
        l.seen = model.seen

    if use_cuda:

        model = model.to(device)
        logging('Use CUDA train only 1 GPU')

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    global optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)

    if continuetrain is not None:
        # print('Continue Train model from ',continuetrain)
        checkpoint = torch.load(continuetrain)
        optimizer.load_state_dict(checkpoint['optimizer'])


    if adaptation > 0:
        freeze_weight_adaptation(adaptation)


    global train_dataset, valid_dataset
    
    
    
    init_width, init_height = 640, 512
    KAIST_dataset = dataset.IDADataset(trainlist, shape=(init_width, init_height), shuffle=True,
                                       transform=transforms.Compose([transforms.ToTensor()]),
                                       train=False, seen=0, batch_size=batch_size,
                                       num_workers=num_workers, condition=condition)
    
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    KAIST_loader = torch.utils.data.DataLoader(KAIST_dataset, batch_size = 1, **kwargs)
    f_min = np.full(k, +10000.0)
    f_max = np.full(k, -10000.0)
    
    for img, name in tqdm.tqdm(KAIST_loader):
        
        img = img.to(device)
        optimizer.zero_grad()

        temp = model(img)
        feature = model.feature
        feature = torch.mean(feature, dim = (2, 3))
        feature = feature.view(k, -1)
        feature = torch.mean(feature, dim = 1)
        feature = feature.cpu().detach().numpy()
        feature_dic[name[0]] = feature
        for i in range(k):
            if feature[i] > f_max[i]:
                f_max[i] = feature[i]
            if feature[i] < f_min[i]:
                f_min[i] = feature[i]
        
        
        del img, temp, feature
        gc.collect()
    
    for key in feature_dic.keys():
        feature_dic[key] = (feature_dic[key] - f_min) / (f_max - f_min)
    
    np.save('visible_feature.npy', feature_dic)
    print('finish save')

        
        









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/kaist_visible.data', help='data definition file')
    parser.add_argument('--config', '-c', type=str, default='cfg/yolov3_kaist.cfg', help='network configuration file')
    # parser.add_argument('--weights', '-w', type=str, default=None, help='initial weights file')
    parser.add_argument('--weights', '-w', type=str, default='backupvisible/yolov3_kaist_000024.weights', help='initial weights file')
    # parser.add_argument('--continuetrain', '-t', type=str, default='backup/adapter_segment_000017.model', help='load model train')
    parser.add_argument('--continuetrain', '-t', type=str, default=None, help='load model train')
    parser.add_argument('--eval', '-n', dest='eval', action='store_true', default=True, help='prohibit test evalulation')
    parser.add_argument('--reset', '-r', action="store_true", default=True, help='initialize the epoch and model seen value')
    parser.add_argument('--epoch', '-e', type=int, default=50,help='How many epoch we train, default is 30')
    parser.add_argument('--layerwise', '-l', type=int, default=0, help='Do layerwise for training on number of layer every epoch')
    parser.add_argument('--adaptation', '-a', type=int, default=0,help='Train adaptation freeze some layers')

    FLAGS, _ = parser.parse_known_args()
    main()