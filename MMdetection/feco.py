import numpy as np
from tqdm import tqdm
import IDA_config

ida = IDA_config.IDA
k = ida.K
N = ida.N

mat = np.full((k, N, N), 0.000001)

thermal_dic = np.load('thermal_feature.npy', allow_pickle = True).item()
visible_dic = np.load('visible_feature.npy', allow_pickle = True).item()


for thermal_name in tqdm(thermal_dic.keys()):
    if ida.dataset == 'KAIST':
        visible_name = thermal_name.replace('lwir', 'visible')
    if ida.dataset == 'FLIR':
        visible_name = thermal_name
    fv = visible_dic[visible_name]
    ft = thermal_dic[thermal_name]
    for i in range(k):
        mat[i, min(int(ft[i] * N), N - 1), min(int(fv[i] * N), N - 1)] += 1.0

mat = mat / np.sum(mat, axis = 2).reshape((k, N, 1))


mean_mat = np.zeros((k, N))
var_mat = np.zeros((k, N))

for i in range(k):
    for j in range(N):
        for p in range(N):
            mean_mat[i, j] += mat[i, j, p] * p / N
            var_mat[i, j] += mat[i, j, p] * ((p / N) ** 2)
var_mat -= mean_mat ** 2

meanvar_dic = {}
for thermal_name in tqdm(thermal_dic.keys()):
    ft = thermal_dic[thermal_name]
    m = np.empty(k)
    v = np.empty(k)
    for i in range(k):
        m[i] = mean_mat[i, min(int(ft[i] * N), N - 1)]
        v[i] = var_mat[i, min(int(ft[i] * N), N - 1)]
    if ida.dataset == 'KAIST':
        meanvar_dic[thermal_name] = [m, v]
    if ida.dataset == 'FLIR':
        meanvar_dic[thermal_name.replace('png', 'jpeg')] = [m, v]
    
 
np.save('meanvar_dic.npy', meanvar_dic)       
            

    
    