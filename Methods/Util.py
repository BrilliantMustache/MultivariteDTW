import math
import os
import numpy as np
import pandas as pd
import glob

import time

import re


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


def DTW_a(s1, s2, windowSize, bestdist):
    # DTW with early abandoning
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
        d=float('inf')
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            if (d>DTW[(i,j)]):
                d = DTW[(i,j)]
        if d>=bestdist:
            return d
    return DTW[len(s1)-1, len(s2)-1]

def DTWDistanceWindowLB_Ordered_xs(i, LBs, DTWdist):
    skips = 0

    start = time.time()
    LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
    dist = DTW(query, references[predId], w)
#    dist = DTWdist[i][predId]   # xs: changed
    for x in range(1,len(LBSortedIndex)):
        if dist>LBs[LBSortedIndex[x]]:
#           Use saved DTW distances from baseline
            dist2 = DTWdist[i][LBSortedIndex[x]]
            if dist>=dist2:
                dist = dist2
                predId = LBSortedIndex[x]
        else:
            skips = skips + 1
    end = time.time()
    coreTime = end - start
    return dist, predId, skips, coreTime

def DTWDistanceWindowLB_Ordered_xs_a(LBs, w, query, references):
    skips = 0

    start = time.time()
    LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
    dist = DTW(query, references[predId], w)
    for x in range(1,len(LBSortedIndex)):
        if dist>LBs[LBSortedIndex[x]]:
#           Use saved DTW distances from baseline
            dist2 = DTW_a(query, references[LBSortedIndex[x]],w, dist)
            #dist2 = DTW(query, references[LBSortedIndex[x]],w)
            if dist>dist2:
                dist = dist2
                predId = LBSortedIndex[x]
        else:
            skips = skips + 1
    end = time.time()
    coreTime = end - start
    return dist, predId, skips, coreTime

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

def get_skips_a (dataset, maxdim, w, lbs, queries, references):
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
        results.append(DTWDistanceWindowLB_Ordered_xs_a(lbs[ids1], w, queries[ids1], references))
    return results


def loadt1dtw(pathUCRResult, maxdim, window):
    '''
    Load the time of one DTW for all datasets
    :param maxdim:
    :param window:
    :return: an nd array with all the times included
    '''
    t1dtwFile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+ '/Any_Anyw'+str(window)+'_t1dtw.npy'
    t1dtw = np.load(t1dtwFile)
    return t1dtw

def loadt1nd (pathUCRResult, maxdim, window):
    '''
    Load the time of one neighbor distance for all datasets.
    :param maxdim:
    :param window:
    :return: an nd array with all the times included
    '''
    t1ndFile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+ '/Any_Anyw'+str(window)+'_t1nd.npy'
    t1nd = np.load(t1ndFile)
    return t1nd

def loadt1bb (pathUCRResult, maxdim, window):
    '''
    Load the time of one neighbor distance for all datasets.
    :param maxdim:
    :param window:
    :return: an nd array with all the times included
    '''
    t1bbFile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+ '/Any_Anyw'+str(window)+'_t1bb.npy'
    t1bb = np.load(t1bbFile)
    return t1bb

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
    if n==0:
        allData = [normalize(pd.read_pickle(g).fillna(0)) for g in glob.glob(path + datasetName + "/*.pkl")]
    else:
        cnt = 0
        allData = []
        pklfiles = glob.glob(path + datasetName + "/*.pkl")
        pklfiles = sorted(pklfiles, key=lambda x: float(re.findall("(\d+)", x)[0]))
        for g in pklfiles:
            cnt +=1
            if (cnt > n):
                break
            else:
                allData.append(normalize(pd.read_pickle(g).fillna(0)))
    return allData

def intlist2str(A):
    return '_'.join([str(a) for a in A])

def load_M0LBs(pathUCRResult, dataset, maxdim, w, nqueries, nreferences):
    lb_2003 = np.load(pathUCRResult+dataset+"/d"+ str(maxdim) +"/w"+ str(w) + '/' +
                      str(nqueries) + "X" + str(nreferences) +"_X0_lbs.npy")
    return lb_2003

def getGroundTruth (dataset, maxdim, w, nqueries, nreferences, pathUCRResult='../Results/UCR/'):
    '''
    Assuming that the DTW distances between all queries and all references are already available. This function
    generates the ground truth of the nearest distance and neighbor for each query. It outputs the results to
    a text file, and an npy file.
    :param dataset:
    :param maxdim:
    :param w:
    :param nqueries:
    :param nreferences:
    :return: 0
    '''
    distanceFileName = pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                       str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
    textoutputFile = pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                       str(nqueries) + "X" + str(nreferences) + "_NoLB_results.txt"
    distances = np.load(distanceFileName)
    minDistances = np.min(distances, axis=1).tolist()
    indices = np.argmin(distances, axis=1).tolist()
    results = zip(minDistances, indices)
    with open(textoutputFile,'w') as f:
        for r in results:
            f.write(str(r) + '\n')
#    np.save(npyoutputFile, np.array(results))

def findErrors (dataset, maxdim, w, nqueries, nreferences, results, pathUCRResult='../Results/UCR/'):
    '''
    Check whether the results are valid.
    :param dataset: the name of the dataset of interest
    :param dim: the maximum dim
    :param w: window size
    :param results: an ndarray [ [dtwDistance, nearest neighbor, etc.] ... ]
    :return: the list of indices that are incorrect
    '''
    errorQueries = []
    resultFile = pathUCRResult+dataset+'/d'+str(maxdim)+'/w'+str(w)+'/' + str(nqueries) + 'X' + \
                          str(nreferences) + '_Z9_results.txt'
    if not os.path.exists(resultFile):
        getGroundTruth(dataset,maxdim,w,nqueries,nreferences,pathUCRResult)
    groundTruth = readResultFile(resultFile)
    results=np.array(results)
    for i in range(groundTruth.shape[0]):
        if (groundTruth[i,0] != results[i,0]):
            errorQueries.append(i)
    return errorQueries

def getGroundTruth_allDataSets (maxdim, windows, nqueries, nreferences, pathUCRResult='../Results/UCR/'):
    datasets = []
    with open(pathUCRResult + "allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    for idxset, dataset in enumerate(datasets):
        print(dataset + " Start!")
        for w in windows:
            getGroundTruth (dataset, maxdim, w, nqueries, nreferences, pathUCRResult)

def readResultFile (f):
    '''
    Read in a result file and store it into an nd array
    :param f: the result file name
    :return: an nd array
    '''
    list = []
    with open(f,'r') as f:
        lines = f.readlines()
        for ln in lines:
            ln = ln.strip().strip("(").strip(")")
            if ln!='':
                list.append([float(a) for a in ln.split(',')])
    return (np.array(list))

#####################################################
# Main Entry
if __name__ == '__main__':
    pathUCRResult = "../Results/UCR/"
    datapath = "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]

#    getGroundTruth_allDataSets (maxdim_g, windows_g, nqueries_g, nreferences_g, pathUCRResult)

    datasets = []
    with open(pathUCRResult + "allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

    for dataset in datasets:
        resultFile = '../Results/UCR/'+dataset + '/d'+str(maxdim_g)+'/w'+str(windows_g[0])+'/' + str(nqueries_g) + \
                     'X' + str(nreferences_g) + '_Xr3K8Q4_results.txt'
        #print(resultFile)
        results = readResultFile(resultFile)
        #results[0,0] = '-3333'
        errs = findErrors(dataset, maxdim_g, windows_g[0], nqueries_g, nreferences_g, results)
        if errs==[]:
            print(dataset + " passed")
        else:
            print(dataset + " error.")
    print("End")
