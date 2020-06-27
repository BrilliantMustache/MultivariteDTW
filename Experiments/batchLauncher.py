# this is the launcher of the batch all series script
# to be executed on a cluster login node
import os


alldatasetsNameFile = "../Results/UCR/workableDatasets.txt"
alldatasetsSizeFile = "../Results/UCR/workableDatasets_size.txt"
with open(alldatasetsNameFile,'r') as f:
    datasets = f.read().strip().split('\n')
with open(alldatasetsSizeFile,'r') as f:
    datasizes = f.read().strip().split('\n')
datasizes = [int(x) for x in datasizes]

for i in range(len(datasets)):
    if (datasizes[i]<1000):
        print(datasets[i]+' full run.')
        os.system("srun -n 32 -N 2 -w c[30-77] -l python3 batchAllSeries " + str(i))
    else:
        print(datasets[i]+' stage_1 run.')
        os.system("srun -n 32 -N 2 -w c[30-77] -l python3 batchAllSeries_stage1 " + str(i))
