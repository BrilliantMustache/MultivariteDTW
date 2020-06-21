import time
from SourceCode.Util import *

# This file implements quantization-based clustering for LB_MV.
# It is the offline candidate clustering version, with adaptive cluster numbers.


def getLBs (dataset, query, reference, w, dim, K=4, Q=2):
    nqueries = len(query)
    length=len(query[0])
    nrefs=len(reference)
    windowSize = w if w <= length / 2 else int(length / 2)
    print("W=" + str(windowSize) + '\n')

    # #  2003/Loose Lower Bounds
    # #  ---------------------------------------------------------------------------------------------
    # print("Starting Loose....")
    #
    # #  Calculate slices range
    # print("Slices Start!")
    # start=time.time()
    # allslices = slice_bounds(query[0], reference, windowSize, dim)
    # end=time.time()
    # setuptime2003=end-start
    # print("Slices Done!")
    #
    # #  Calculate loose Lower Bounds
    # print("Loose Start!")
    # start=time.time()
    # lbs_2003 = [multLB_2003(query[ids1], reference, dim, allslices) for ids1 in range(len(query))]
    # end=time.time()
    # lbtime2003=end-start
    # np.save(pathUCRResult+"" + dataset + '/' + str(w) + "/"+str(nqueries)+"X"+str(nrefs)+ "_2003_lbs.npy", lbs_2003)
    # print("Loose Done!" + '\n')

    # cluster-2003 Lower Bounds
    # ---------------------------------------------------------------------------------------------
    print("Starting cluster-2003-quick ....")
    #  Calculate slices range
    print("Bounding boxes finding Start!")
    start=time.time()
    bboxes = [findBoundingBoxes(np.array(ref), K, windowSize, Q) for ref in reference]
    end=time.time()
    setuptime2003cluster_q=end-start
    print("Bounding boxes Done!")

    #  Calculate Lower Bounds
    print("Cluster-2003-quick lower bounds. Start!")
    start=time.time()
    lbs_2003_cluster_q = [getLB_oneQ (query[ids1], reference, dim, bboxes) for ids1 in range(len(query))]
    end=time.time()
    lbtime2003cluster_q=end-start
    # np.save(pathUCRResult+ dataset + "/" + str(w) + "/" + str(nqueries_g) + "X" +
    #         str(nreferences_g) + "_M3"+ "K"+ intlist2str(K) +"Q" + intlist2str(Q) + "_lbs.npy", lbs_2003_cluster_q)
    print("Cluster-2003-quick Done!" + '\n')

#    thistimes = [setuptime2003cluster_q, lbtime2003cluster_q]

#    np.save(pathUCRResult+ dataset + "/" + str(w) + "/" + str(nqueries_g) + "X" +
#            str(nreferences_g) + "_M3"+ "K"+ intlist2str(K) +"Q" + intlist2str(Q) + "_times.npy", thistimes)

#    allTimes_g.append([setuptime2003cluster_q, lbtime2003cluster_q])

    return lbs_2003_cluster_q, [setuptime2003cluster_q, lbtime2003cluster_q]

def findBoundingBoxes (ref, K, W, Q):
    '''
    find the K bounding boxes for each window in ref with quantizations
    :param ref: a data frame holding a reference series
    :param K: the number of bounding boxes
    :param W: the window size
    :param Q: the number of cells in each dimension
    :return: a len(ref)*K array with each element [ [dim low ends] [dim high ends]]
    '''
    length = ref.shape[0]
    dims = ref.shape[1]
    allBoxes = []
    for idx in range(length):
#        nonEmptyCells = {}
        cellMembers = {}
        bboxes = []
        awindow = ref[(idx - W if idx - W >= 0 else 0):(idx + W if idx + W <= length else length)]
        overall_ls = [min(np.array(awindow)[:,idd]) for idd in range(dims)]
        overall_us = [max(np.array(awindow)[:, idd]) for idd in range(dims)]
        cells = [1+int((overall_us[idd] - overall_ls[idd])*Q) for idd in range(dims)]
        celllens = [ (overall_us[idd] - overall_ls[idd])/cells[idd]+0.00000001 for idd in range(dims) ]
        for e in awindow:
            thiscell = str([int( (e[idd]-overall_ls[idd])/celllens[idd]) for idd in range(dims)])
            if thiscell in cellMembers:
                cellMembers[thiscell].append(e)
            else:
                cellMembers[thiscell] = [e]
#        if len(cellMembers.keys())>K:
#            bboxes=[[overall_ls, overall_us]]
#        else:
        for g in cellMembers:
            l = [min(np.array(cellMembers[g])[:, idd]) for idd in range(dims)]
            u = [max(np.array(cellMembers[g])[:, idd]) for idd in range(dims)]
            bboxes.append([l, u])
        if len(bboxes)>K:
            # combine all boxes except the first K-1 boxes
            sublist = bboxes[K-1:]
            combinedL = [min([ b[0][idd] for b in sublist]) for idd in range(dims)]
            combinedU = [max([b[1][idd] for b in sublist]) for idd in range(dims)]
            bboxes= bboxes[0:K-1]
            bboxes.append([combinedL, combinedU])
        allBoxes.append(bboxes)
    return np.array(allBoxes)


def getLB_oneQ (X, others, dim, sl_bounds):
    #  X is one series, others is all references, dim is dimensions, sl_bounds has all the bounding boxes of all reference series
    lbs = []
    for idy, s2 in enumerate(others):
        temps = []
        LB_sum = 0
        slboundsOneY = sl_bounds[idy]
        for idx, x in enumerate(X):
            numBoxes = len(slboundsOneY[idx])
            oneYbounds=[]
            for idbox in range(numBoxes):
                l = slboundsOneY[idx][idbox][0]
                u = slboundsOneY[idx][idbox][1]
                temp = math.sqrt(sum([(x[idd]-u[idd]) ** 2 if (x[idd] > u[idd]) else (l[idd]-x[idd])**2 if (x[idd] < l[idd]) else 0
                               for idd in range(dim)]))
                oneYbounds.append(temp)
            LB_sum+=min(oneYbounds)
        lbs.append(LB_sum)
    return lbs


def loadSkips (datasets, maxdim, windowSizes, nqueries, nrefs, Ks, Qs):
    skips_all = []
    for dataset in datasets:
        for idx, w in enumerate(windowSizes):
            skips_temp = []
            for K in Ks:
                for Q in Qs:
                    with open(pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"+str(nqueries)+"X"+
                                      str(nrefs)+ "_M3K"+str(K)+"_Q"+str(Q)+"_results.txt", 'r') as f:
                        temp = f.readlines()
                        temps = [l.strip()[1:-1] for l in temp]
                        results = [t.split(',') for t in temps]
                        skips = [int(r[2]) for r in results]
                        skips_temp.append(sum(skips))
            skips_all.append(skips_temp)
    return skips_all

def dataCollection(datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2]):
    datasets=[]
    with open(pathUCRResult+"allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    with open(pathUCRResult+"size_no_EigenWorms.txt",'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

    # # create directories if necessary
    # for datasetName in datasets:
    #     for w in windows:
    #         dir = pathUCRResult+"" + datasetName + "/" + str(w)
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)

    #times = np.load(pathUCRResult+"" + '/' + str(nqueries) + "X" + str(nreferences) + "_times_2003_cluster.npy")

    # get # of skips quickly
    # for datasetName in datasets:
    #     for w in windows:
    #         lb_quant, lb_2003 = load_saved_lb(datasetName, w)
    #         get_skips_quick(datasetName, w, lb_quant, lb_2003, 3)
    # print("get_skips_quick is done.")

    ################
    allTimes = []
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
            for K in Ks:
                for Q in Qs:
                    print("K="+str(K)+" Q="+str(Q))
                    lbs_M3, times = getLBs (dataset, query, reference, w, dim, K, Q)
                    np.save(pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
                            + str(nqueries) + "X" + str(nreferences) + "_M3K" + str(K) + "Q" + str(Q) + "_lbs.npy", lbs_M3)
                    allTimes.append(times)
                    results = get_skips(dataset, maxdim, w, lbs_M3, query, reference)
                    with open(pathUCRResult + dataset + '/' + 'd' + str(maxdim) + '/w' + str(w) + "/" + str(
                            nqueries) + "X" + str(
                            nreferences) + "_" + "M3K" + str(K) + "Q" + str(Q) + "_results" + ".txt", 'w') as f:
                        for r in results:
                            f.write(str(r) + '\n')
        print(dataset+" Done!"+'\n'+'\n')

    np.save(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" + str(nqueries)+"X"+str(nreferences)
            + "_M3"+"w" + intlist2str(windows)+ "K"+intlist2str(Ks)+"Q"+intlist2str(Qs) + "_times.npy", allTimes)

def dataProcessing(maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2]):
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

    # get times
    allM3Times  = np.load(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" + str(nqueries)+"X"+str(nreferences)
            + "_M3"+"w" + str(windows)+ "K"+intlist2str(Ks)+"Q"+intlist2str(Qs) + "_times.npy")
    #  [ dataset1[ [setupTime LBTime] ... ] ... ]
    allM3Times = allM3Times.reshape((-1, 2*len(Ks)*len(Qs)))
    datasetsNum = allM3Times.shape[0]
    M3Settings = int(allM3Times.shape[1]/2)
    M3SetupTimes = [ [allM3Times[d][s*2] for s in range(M3Settings)] for d in range(datasetsNum)]
    M3LBTimes = [[allM3Times[d][s*2+1] for s in range(M3Settings)] for d in range(datasetsNum)]

    # [ [data1_SetupTime data1_LBTime] [data2_SetupTime data2_LBTime] ... ]
    allM0Times = np.load(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" +str(nqueries)+"X"+str(nreferences)
            + "_M0"+"w" + str(windows)+ "_times.npy")
    M0SetupTimes = [ allM0Times[d][0] for d in range(datasetsNum)]
    M0LBTimes = [allM0Times[d][1] for d in range(datasetsNum)]

    M3SetupRatios = [[M3SetupTimes[d][s] / M0LBTimes[d] for s in range(M3Settings)] for d in range(len(datasets))]
    M3LBRatios = [ [M3LBTimes[d][s]/M0LBTimes[d] for s in range(M3Settings)] for d in range(len(datasets))]

    # get skips
    # skips_all: 2003, M2, M3_setting1, M3_setting2, ...
    skips_all = loadSkips(datasets, maxdim, windows, nqueries, nreferences, Ks, Qs)
    M3Skips = np.array(skips_all)
    #    M0Skips = np.array(skips_all)[:,0]
    #    np.save(pathUCRResult+"UsedForPaper/" + str(nqueries) + "X" + str(nreferences) + "M0skips.npy", M0Skips)

    # save all the data to files
    np.save(pathUCRResult+"UsedForPaper/"+str(nqueries) + "X" + str(nreferences) + "M3SetupRatios.npy", M3SetupRatios)
    np.save(pathUCRResult+"UsedForPaper/" + str(nqueries) + "X" + str(nreferences) + "M3LBRatios.npy",
            M3LBRatios)
    np.save(pathUCRResult+"UsedForPaper/"+str(nqueries) + "X" + str(nreferences) + "M3skips.npy", M3Skips)

    print("data saved.")

#####################################################
# Main Entry
if __name__ == '__main__':
    pathUCRResult = "../Results/UCR/"
    datapath = "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    Ks_g = [4, 6, 8]
    Qs_g = [2, 3, 4]
    #windows_g = [20]
    windows_g = [20]
    dataCollection(datapath, maxdim_g,nqueries_g,nreferences_g,windows_g,Ks_g,Qs_g)
    #dataProcessing(maxdim_g,nqueries_g,nreferences_g,windows_g,Ks_g,Qs_g)

    print("End")