import os
import glob
import re
import numpy as np

def myrename(oldpattern, newpattern, methodName):
    for dataset in datasets_g:
        datasetpath = pathUCRResults_g + dataset + '/'
        dirs = [x[0] for x in os.walk(datasetpath)]
        for d in dirs[1:]:
            fullpattern = d + "/*" + oldpattern
            oldFiles = glob.glob(fullpattern)
            for f in oldFiles:
                if not methodName in f:
                    nf = f.replace(oldpattern, newpattern)
                    os.rename(f, nf)

def myrename_KQ_lbs_times(oldpattern, newpattern, methodName, suffix, keywords,content):
    for dataset in datasets_g:
        datasetpath = pathUCRResults_g + dataset + '/'
        dirs = [x[0] for x in os.walk(datasetpath)]
        for d in dirs[1:]:
            fullpattern = d + "/*" + oldpattern +'*'+suffix
            oldFiles = glob.glob(fullpattern)
            for f in oldFiles:
                if not methodName in f:
                    m=re.search(keywords+'(.?)_Q(.?)',f)
                    if m:
                        k=m.group(1)
                        q=m.group(2)
                        nf = f.replace(oldpattern+k+'_Q'+q, newpattern+'K'+k+'Q'+q+"_"+content)
                        os.rename(f, nf)


def myrename_KQ_results(oldpattern, newpattern, methodName, suffix, keywords,content):
    for dataset in datasets_g:
        datasetpath = pathUCRResults_g + dataset + '/'
        dirs = [x[0] for x in os.walk(datasetpath)]
        for d in dirs[1:]:
            fullpattern = d + "/*" + oldpattern +'*'+suffix
            oldFiles = glob.glob(fullpattern)
            for f in oldFiles:
                if not methodName in f:
                    m=re.search(keywords+'(.?)_Q(.?)',f)
                    if m:
                        k=m.group(1)
                        q=m.group(2)
                        nf = f.replace(oldpattern+'_K'+k+'_Q'+q, newpattern+'K'+k+'Q'+q+"_"+content)
                        os.rename(f, nf)

def copyDistanceFilesOver(datasets):
    oldpath = "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Results/UCR/"
    oldName = "3X20_DTWdistances.npy"
    newpath = "../../Results/UCR/"
    newsubpath = "d5/w20/"
    newName = "3X20_NoLB_DTWdistances.npy"

    for d in datasets:
        oldfile = oldpath+d+'/20/'+oldName
        newfile = newpath+d+'/'+newsubpath+newName
        dist = np.load(oldfile)
        np.save(newfile, dist)

datasets_g = []
pathUCRResults_g = "../../Results/UCR/"
with open(pathUCRResults_g+ "/allDataSetsNames_no_EigenWorms.txt", 'r') as f:
#with open("../Results/UCR/allDataSetsNames_no_EigenWorms.txt", 'r') as f:
    for line in f:
        datasets_g.append(line.strip())
f.close()

copyDistanceFilesOver(datasets_g)
exit()

# rename all DTW distance files
oldpattern_g = "_DTWdistances.npy"
newpattern_g = '_NoLB_DTWdistances.npy'
methodName_g = '_NoLB_'
myrename(oldpattern_g, newpattern_g, methodName_g)


# rename 2003_lbs to M0_lbs
oldpattern_g = "_2003_lbs.npy"
newpattern_g = "_M0_lbs.npy"
methodName_g = '_M0_'
myrename(oldpattern_g, newpattern_g, methodName_g)


# rename 2003_results to M0_results
oldpattern_g = "_2003_results.txt"
newpattern_g = "_M0_results.txt"
methodName_g = '_M0_'
myrename(oldpattern_g, newpattern_g, methodName_g)

# rename 2003cluster_quick_lbsX_QY.npy to M3KXQY_lbs.npy
oldpattern_g = "_2003cluster_quick_lbs"
newpattern_g = "_M3"
methodName_g = '_M3'
suffix = '.npy'
myrename_KQ_lbs_times(oldpattern_g, newpattern_g, methodName_g,suffix,'_lbs','lbs')


# rename 2003_results to M0_results
oldpattern_g = "_2003cluster_quick_results"
newpattern_g = "_M3"
methodName_g = '_M3'
suffix='.txt'
myrename_KQ_results(oldpattern_g, newpattern_g, methodName_g,suffix,'_results_K','results')


# rename 2003_results to M0_results
oldpattern_g = "_2003cluster_quick_times"
newpattern_g = "_M3"
methodName_g = '_M3'
suffix='.npy'
myrename_KQ_lbs_times(oldpattern_g, newpattern_g, methodName_g,suffix,'_times','times')

# modify the folder structures
for d in datasets_g:
    os.makedirs(pathUCRResults_g+d+'/d5'+'/w20')
