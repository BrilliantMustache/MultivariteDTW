import math
import os
import numpy as np
import pandas as pd
import glob

def calNeighborDistances(A):
    aa = [distance(A[i,:], A[i+1,:]) for i in range(0, len(A)-1)]
    return aa

def distance(p1, p2):
    x = 0
    for i in range(len(p1)):
        x += (p1[i] - p2[i]) ** 2
    return math.sqrt(x)

def DTW(s1, s2, windowSize):
    DTW = {}
    w = max(windowSize, abs(len(s1)-len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return DTW[len(s1)-1, len(s2)-1]

def DTWDistanceWindowLB_Ordered_xs(i, LBs, DTWdist):
    skip = 0

    LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
    dist = DTWdist[i][predId]   # xs: changed

    for x in range(1,len(LBSortedIndex)):
        if dist>LBs[LBSortedIndex[x]]:  # xs: changed
#           Use saved DTW distances from baseline
            dist2 = DTWdist[i][LBSortedIndex[x]]
            if dist>=dist2:
                dist = dist2
                predId = LBSortedIndex[x]
        else:
            skip = skip + 1

    return dist, predId, skip

def get_skips (dataset, maxdim, w, lbs, queries, references):
    nqueries=len(queries)
    nrefs=len(references)
    print("W="+str(w)+'\n')
    distanceFileName = "../Results/UCR/" + dataset + '/d' + str(maxdim) + '/w'+ str(w) + "/"+str(nqueries)\
                       +"X"+str(nrefs)+"_NoLB_DTWdistances.npy"
    if not os.path.exists(distanceFileName):
        distances = [[DTW(s1, s2, w) for s2 in references] for s1 in queries]
        np.save(distanceFileName,np.array(distances))
    else:
        distances = np.load(distanceFileName)

    results =[]
    for ids1 in range(nqueries):
        results.append(DTWDistanceWindowLB_Ordered_xs(ids1, lbs[ids1], distances))

    return results

def DTWwlb(s1,s2,hwindowSize):
    '''
    Compute DTW between q and r and also the tight lower bound between them
    :param s1: query series
    :param s2: reference series
    :param hwindowSize: half window size
    :return: dtw distance, tight lower bound
    '''
    DTW = {}
    w = max(hwindowSize, abs(len(s1)-len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')

    DTW[(-1, -1)] = 0

    lb = 0
    for i in range(len(s1)):
        mn=float("inf")
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            mn=dist if dist<mn else mn
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
        lb+=mn
    return DTW[len(s1)-1, len(s2)-1], lb

def edist(A,B):
    '''
    Compute the enclean distance between two np arrays
    :param A: an np array
    :param B: an np array
    :return: their enclean distance
    '''
    return sum([math.sqrt(sum( [(A[i][j]-B[i][j])**2 for j in range(A.shape[1])] )) for i in range(A.shape[0])])

def normalize(aserie):
    # data structure: DataFrame [ dim1 [...], dim2 [...], ...] ]
    nmSerie = []
    for d in range(aserie.shape[1]):
        oneDim = list(aserie[d])
        mi = min(oneDim)
        ma = max(oneDim)
        dif = (ma - mi) if (ma-mi)>0 else 0.0000000001
        nmValue = [(x-mi)/dif for x in oneDim]
        nmSerie.append(nmValue)
    return pd.DataFrame(nmSerie).transpose()

def loadUCRData_norm_xs (path, name, n):
    dataDims = pd.read_csv(path + "DataDimensions.csv")
    dataDims = dataDims.drop(columns=dataDims.columns[10:])
    dataDims.at[23, "Problem"] = "PhonemeSpectra"
    dataDims = dataDims.set_index('Problem')
    dataDims['Total Instances'] = [(datafile[1] + datafile[2]) * datafile[3] * datafile[4] for id, datafile in dataDims.iterrows()]

    datasetName = name
    if n<0:
        allData = [normalize(pd.read_pickle(g).fillna(0)) for g in glob.glob(path + datasetName + "/*.pkl")]
    else:
        cnt = 0
        allData = []
        for g in glob.glob(path + datasetName + "/*.pkl"):
            cnt +=1
            if (cnt > n):
                break
            else:
                allData.append(normalize(pd.read_pickle(g).fillna(0)))
    return allData

def intlist2str(A):
    return '_'.join([str(a) for a in A])

#global datasetName
