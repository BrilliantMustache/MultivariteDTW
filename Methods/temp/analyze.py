import numpy as np
from Methods.Util import *

path = "../../Results/UCR/_AllDataSets/d5/"
nqueries = 3
nreferences = 20
npairs = nqueries*nreferences

####
maxdim_g = 5
nqueries_g = 3
nreferences_g = 20
windows_g = [20]
machineRatios = [1, 1]
THs_g = [0.05, 0.1, 0.2]
Ks_g = [4, 6, 8]
Qs_g = [2, 3, 4]
THs_g = [0.5]
K_g = 4
period_g = 5

x3rseac_spd = np.load('/Users/xshen/PycharmProjects/MultivariteDTW/Results/UCR/_AllDataSets/d5/3X20_X3_rseac_w20K6_8Q2_3C0_speedups.npy')

nboxes = np.load("/Users/xshen/PycharmProjects/MultivariteDTW/Results/UCR/XipengShenMacOther.local_17/_AllDatasets/d5/0X0_X3_rsea_w20K6Q2_3_nboxes.npy")

Z1ea_chosenSettings = np.load("../../Results/UCR/" + "_AllDataSets/" + 'd' + str(maxdim_g) + "/3X20_Z1_ea_w20TH0.05_0.1_0.2_speedups.npy")
                                                                                             #"/3X20_Z1_ea_w20TH0.2_0.4_0.6_0.8_settingchosen.npy")
Z3ea_chosenSettings = np.load("../../Results/UCR/" + "_AllDataSets/" + 'd' + str(maxdim_g) + '/' + str(nqueries_g) + "X" + str(nreferences_g) +
            "_Z3_ea_w" + str(windows_g[0]) + "K" + intlist2str(Ks_g) + "Q" + intlist2str(Qs_g) + "TH" + intlist2str(THs_g) + '_settingchosen.npy')
###
z0aSpd = np.load('/Users/xshen/temp/MultivariteDTW/Results/fas10/_AllDataSets/d5/3X20_Z0_a_w20_speedups.npy')

speedups = np.genfromtxt(path+'3X20_All_speedups.txt', delimiter=',')
overhead = np.genfromtxt(path+'3X20_All_overhead.txt', delimiter=',')
skips = np.genfromtxt(path+'3X20_All_skips.txt', delimiter=',')
originalt1dtw = np.load(path + '/' + 'Any_Anyw20' + '_t1dtw.npy')

speedups0 = speedups[:,0]
#overhead0 = overhead[:,0]
skips0 = skips[:,0]
speedup3 = speedups[:,20]
#overhead3 = overhead[:,20]
skips3 = skips[:,20]


dtwtime0 = np.load(path+'3X20_X0_a_w20_dtwtime.npy')
t1dtw = dtwtime0/(npairs-skips0)

ttlTime = originalt1dtw*npairs/speedup3
overhead3per = (ttlTime - t1dtw*skips3)/(npairs - skips3)


print('done')