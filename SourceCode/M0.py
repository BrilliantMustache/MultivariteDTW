import time
from SourceCode.Util import *

# This file implements the LB_MV method (proposed in 2003 for multivariate DTW
# The dataCollection function saves the following:
#     the lower bounds in each individual directory: an nd array
#     the DTW distances and skips in each individual directory: a test file
#     the setup time and total lower bound time of each dataset in one overall file in AllDataSets directory: an nd array

def slice_bounds (X, others, W, dim):
    slices = []
    for s2 in others:
        temp = []
        s2_l_1 = len(s2) - 1
        for idx in range(len(X)):
            s2_slice = s2[(idx - W if idx - W >= 0 else 0):(idx + W + 1 if idx + W <= s2_l_1 else s2_l_1+1)]
            l = [min(s2_slice[:, idd]) for idd in range(dim)]
            u = [max(s2_slice[:, idd]) for idd in range(dim)]
            temp.append([l,u])
        slices.append(temp)
    return slices


def getLB_oneQ (X, others, dim, sl_bounds):
    '''
    Get the lower bounds between one query series X and many candidate series in others
    :param X: one series
    :param others: all candidate series
    :param dim: dimension of a point
    :param sl_bounds: the bounding boxes of the candidate windows
    :return: the lower bounds between X and each candidate series
    '''
    lbs = []
    for idy, s2 in enumerate(others):
        temps = []
        LB_sum = 0
        sl = sl_bounds[idy]
        for idx, x in enumerate(X):
            l = sl[idx][0]
            u = sl[idx][1]
            temp = math.sqrt(sum([(x[idd]-u[idd]) ** 2 if (x[idd] > u[idd]) else (l[idd]-x[idd])**2 if (x[idd] < l[idd]) else 0
                           for idd in range(dim)]))
            temps.append(temp)
            LB_sum+=temp
        lbs.append(LB_sum)
    return lbs

def getLBs (dataset, query, reference, w, dim):
    nqueries = len(query)
    length = len(query[0])
    nrefs=len(reference)
    windowSize = w if w <= length / 2 else int(length / 2)
    print("W=" + str(windowSize) + '\n')

    print("Starting Loose....")

    #  Calculate slices range
    print("Slices Start!")
    start=time.time()
    allslices = slice_bounds(query[0], reference, windowSize, dim)
    end=time.time()
    setuptime2003=end-start
    print("Slices Done!")

    #  Calculate loose Lower Bounds
    print("Lower bounds start!")
    start=time.time()
    lbs_2003 = [getLB_oneQ(query[ids1], reference, dim, allslices) for ids1 in range(len(query))]
    end=time.time()
    lbtime2003=end-start
    print("Lower bounds done!" + '\n')

    return lbs_2003, [setuptime2003, lbtime2003]

def dataCollection(datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20]):
    datasets=[]
    #with open("Results/UCR/allDataSetsNames.txt",'r') as f:
    with open(pathUCRResult+"allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    #with open("Results/UCR/size.txt",'r') as f:
    with open(pathUCRResult+"size_no_EigenWorms.txt",'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()
    # datasets=["ArticularyWordRecognition","AtrialFibrillation"]

    # # create directories if necessary
    # for datasetName in datasets:
    #     for w in windows:
    #         dir = pathUCRResult+"" + datasetName + "/" + str(w)
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)

    allTimes=[]
    for idxset, dataset in enumerate(datasets):
        print(dataset+" Start!")
        assert(datasize[idxset]>=nqueries+nreferences)
        stuff = loadUCRData_norm_xs(datapath, dataset,nqueries+nreferences)
        size = len(stuff)
        length = stuff[0].shape[0]
        dim = min(stuff[0].shape[1], maxdim)
        print("Size: "+str(size))
        print("Dim: "+str(dim))
        print("Length: "+str(length))
        samplequery = stuff[:nqueries]
        samplereference = stuff[nqueries:nreferences+nqueries]

        print(dataset+":  "+ str(nqueries)+" queries, "+ str(nreferences)+ " references." +
              " Total dtw: "+str(nqueries*nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            lbs_2003, times = getLBs (dataset, query, reference, w, dim)
            np.save(pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
                    + str(nqueries) + "X" + str(nreferences) + "_M0_lbs.npy", lbs_2003)
            allTimes.append(times)
            results=get_skips (dataset, maxdim, w, lbs_2003, query, reference)
            with open(pathUCRResult + dataset + '/' + 'd' + str(maxdim) + '/w'+ str(w) + "/" + str(nqueries) + "X" + str(
                    nreferences) + "_" + "M0" + "_results" + ".txt", 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')

        print(dataset+" Done!"+'\n'+'\n')

    np.save(pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/"+ str(nqueries) + "X" + str(nreferences)
            + "_M0w" + intlist2str(windows) + "_times.npy", allTimes)

def loadSkips (datasets, maxdim, windowSizes, nqueries, nrefs):
    skips_all = []
    for dataset in datasets:
        for idx, w in enumerate(windowSizes):
            with open(pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"+
                              str(nqueries)+"X"+str(nrefs)+ "_M0" + "_results.txt", 'r') as f:
                temp = f.readlines()
                temps = [l.strip()[1:-1] for l in temp]
                results = [t.split(',') for t in temps]
                skips = [int(r[2]) for r in results]
                skips_all.append(skips)
    return skips_all

def dataProcessing(maxdim = 5, nqueries = 3, nreferences = 20, windows = [20]):
    datasets=[]
    #with open(pathUCRResult+"allDataSetsNames.txt",'r') as f:
    with open(pathUCRResult+"allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    #with open(pathUCRResult+"size.txt",'r') as f:
    with open(pathUCRResult+"size_no_EigenWorms.txt",'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()
    # datasets=["ArticularyWordRecognition","AtrialFibrillation"]

    skips_all = loadSkips(datasets, maxdim, windows, nqueries, nreferences)
    np.save(pathUCRResult+"UsedForPaper/" + str(nqueries) + "X" + str(nreferences) + "M0skips.npy", skips_all)

########################
# collect M0's data
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    allTimes_g = []
    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    dataCollection(datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
    dataProcessing(maxdim_g, nqueries_g, nreferences_g, windows_g)
    print("End")