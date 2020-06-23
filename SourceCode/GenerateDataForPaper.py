import numpy as np
import pandas as pd
import glob
import os

import scipy


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

path='../Results/UCR/'
maxdim_g = 5
nqueries_g = 3
nreferences_g = 20
windows_g = [20]
allTimes_g = []
Ks_g = [4, 6, 8]
Qs_g = [2, 3, 4]
THs_g = [0.05,0.1,0.2]

datasets = []
with open(path+"allDataSetsNames_no_EigenWorms.txt", 'r') as f:
    for line in f:
        datasets.append(line.strip())
f.close()

datasets = ["ArticularyWordRecognition", "AtrialFibrillation"]

# Write the results out
#table=np.array([M0Speedup, M1Speedup_b, M3Speedup_b, sM3Speedup, M2Speedup, M0Skips/totalPairs,
#                M1Skips_b/totalPairs, M3Skips_b/totalPairs, sM3Skips/totalPairs , M2Skips/totalPairs,
#                M0LBRate, M1LBRate_b, M3LBRate_b, sM3LBRate, M2LBRate]).transpose()
#averageRow = [geo_mean(table[:,i]) for i in range(table.shape[1])]

#methodnames=['X0', 'X1', 'X2', 'X3', 'Xr3', 'Xs3', 'Xz3', 'Z0', 'Z1', 'Z3']
#methods=len(methodnames)

fnm = path+'_AllDataSets/d'+str(maxdim_g)+'/'+ str(nqueries_g)+'X'+str(nreferences_g)
speedupFiles = glob.glob(fnm+"_*_speedups*.npy")
speedupFiles.sort()
skipFiles = glob.glob(fnm+'_*_skips*.npy')
skipFiles.sort()
overheadFiles = glob.glob(fnm+'_*_overheadrate*.npy')
overheadFiles.sort()

speedups = np.array([np.load(f) for f in speedupFiles])
skips = np.array([np.load(f) for f in skipFiles])
overhead = np.array([np.load(f) for f in overheadFiles])
alltable = np.array([speedups, skips, overhead])

print('Done.')
#%%%%%%%%%%%%%%%%%%%%%%%%%%


# with open("Results/UCR/UsedForPaper/OverallPerformanceTable.txt", 'w') as f:
#     boldnum = '{{\\bf {0:.2f}}}\t&'
#     f.write("""\\begin{table*}
#     \\centering
#     \\caption{Overall Performance Comparison}
#     \\label{tab:overall}
#     """)
#     f.write('\\begin{{tabular}}{{|r||*{{{0}}}{{c|}}c||*{{{0}}}{{c|}}c||*{{{0}}}{{c|}}c|}}\\hline\n'.format(methods-1))
#     f.write("Dataset & \\multicolumn{{{0}}}{{c||}}{{Speedups}} & \\multicolumn{{{0}}}{{c||}}{{Skips}} & \\multicolumn{{{0}}}{{c|}}{{LB Overhead}}\\\\\\cline{{2-{1}}}\n".format(methods,methods*3+1))
#     for i in [1,2,3]:
#         for m in methodnames:
#             f.write('& {0} '.format(m))
#     f.write('\\\\\\hline\n')
#     for idx, dataset in enumerate(datasets):
#         f.write(dataset+'\t&')
#         f.write('{0:.2f}\t&'.format(M0Speedup[idx]))
#
#         if M1Speedup_b[idx]>=M0Speedup[idx]:
#             f.write(boldnum.format(M1Speedup_b[idx]))
#         else:
#             f.write('{0:.2f}\t&'.format(M1Speedup_b[idx]))
#
#         if M3Speedup_b[idx]>=M0Speedup[idx]:
#             f.write(boldnum.format(M3Speedup_b[idx]))
#         else:
#             f.write('{0:.2f}\t&'.format(M3Speedup_b[idx]))
#
#         if sM3Speedup[idx]>=M0Speedup[idx]:
#             f.write(boldnum.format(sM3Speedup[idx]))
#         else:
#             f.write('{0:.2f}\t&'.format(sM3Speedup[idx]))
#
#         if M2Speedup[idx]>=M0Speedup[idx]:
#             f.write(boldnum.format(M2Speedup[idx]))
#         else:
#             f.write('{0:.2f}\t&'.format(M2Speedup[idx]))
#
#         for e in table[idx,methods:-1]:
#             f.write('{0:.2f}\t&'.format(e))
#         f.write('{0:.2f}\\\\ \\hline\n'.format(table[idx,-1]))
#     f.write('{{\\bf Average}}\t&')
#     for e in range(len(averageRow)-1):
#         f.write('{0:.2f}\t&'.format(averageRow[e]))
#     f.write('{0:.2f}\\\\ \hline\n'.format(averageRow[-1]))
#     f.write('\end{tabular}\n')
#     f.write('\end{table*}\n')
#     f.close()
#np.savetxt("Results/UCR/UsedForPaper/OverallPerformanceTable.txt", table, fmt=["%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f"], delimiter=' & ')


print("pause here.")
