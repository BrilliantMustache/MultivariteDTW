from Methods.Util import *

# This file implements quantization-based clustering for LB_MV.
# It is the offline candidate clustering version, with adaptive cluster numbers.
# It does not use X0 at all.

# The dataCollection function saves the following:
#     the lower bounds in each individual directory: an nd array
#     the DTW distances and skips and coreTime in each individual directory: a text file
#     the setup time and total lower bound time of each dataset in one overall file in AllDataSets directory: an nd array


def getLBs (dataset, query, reference, w, dim, K=4, Q=2):
    nqueries = len(query)
    length=len(query[0])
    nrefs=len(reference)
    windowSize = w if w <= length / 2 else int(length / 2)
    print("W=" + str(windowSize) + '\n')

    #  Calculate slices range
    print("Bounding boxes finding Start!")
    start=time.time()
    bboxes = [findBoundingBoxes(np.array(ref), K, windowSize, Q) for ref in reference]
    end=time.time()
    nboxes=0
    for r in range(len(reference)):
        boxes = bboxes[r]
        nboxes += sum([len(p) for p in boxes])
    setuptime2003cluster_q=end-start
    print("Bounding boxes Done!")

    #  Calculate Lower Bounds
    print("Cluster-2003-quick lower bounds. Start!")
    start=time.time()
    lbs_2003_cluster_q = [getLB_oneQ (query[ids1], reference, dim, bboxes) for ids1 in range(len(query))]
    end=time.time()
    lbtime2003cluster_q=end-start
    # np.save(pathUCRResult+ dataset + "/" + str(w) + "/" + str(nqueries_g) + "X" +
    #         str(nreferences_g) + "_X3_"+ "K"+ intlist2str(K) +"Q" + intlist2str(Q) + "_lbs.npy", lbs_2003_cluster_q)
    print("Cluster-2003-quick Done!" + '\n')

#    thistimes = [setuptime2003cluster_q, lbtime2003cluster_q]

#    np.save(pathUCRResult+ dataset + "/" + str(w) + "/" + str(nqueries_g) + "X" +
#            str(nreferences_g) + "_X3_"+ "K"+ intlist2str(K) +"Q" + intlist2str(Q) + "_times.npy", thistimes)

#    allTimes_g.append([setuptime2003cluster_q, lbtime2003cluster_q])

    return lbs_2003_cluster_q, [setuptime2003cluster_q, lbtime2003cluster_q], nboxes

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

def dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2]):
    datasets = []
    # with open("Results/UCR/allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize = []
    # with open("Results/UCR/size.txt",'r') as f:
    with open(datasetsSizeFile, 'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

#    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

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
    allnboxes = []
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
        # -------------------------------------------------
        if (nqueries * nreferences == 0):  # all series to be used
            qfrac = 0.3
            samplequery = stuff[:int(size * qfrac)]
            samplereference = stuff[int(size * qfrac):]
        # -------------------------------------------------

        print(dataset+":  "+ str(nqueries)+" queries, "+ str(nreferences)+ " references." +
              " Total dtw: "+str(nqueries*nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            for K in Ks:
                for Q in Qs:
                    print("K="+str(K)+" Q="+str(Q))
                    lbs_X3, times, nboxes = getLBs (dataset, query, reference, w, dim, K, Q)
                    allnboxes.append(nboxes)
                    np.save(pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
                            + str(nqueries) + "X" + str(nreferences) + "_X3_K" + str(K) + "Q" + str(Q) + "_lbs.npy", lbs_X3)
                    allTimes.append(times)
                    results = get_skips(dataset, maxdim, w, lbs_X3, query, reference, pathUCRResult)
                    if findErrors(dataset, maxdim, w, nqueries, nreferences, results, pathUCRResult):
                        print('Wrong Results!! Dataset: ' + dataset)
                        exit()
                    with open(pathUCRResult + dataset + '/' + 'd' + str(maxdim) + '/w' + str(w) + "/" + str(
                            nqueries) + "X" + str(
                            nreferences) + "_" + "X3_K" + str(K) + "Q" + str(Q) + "_results" + ".txt", 'w') as f:
                        for r in results:
                            f.write(str(r) + '\n')
        print(dataset+" Done!"+'\n'+'\n')

    np.save(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" + str(nqueries)+"X"+str(nreferences)
            + "_X3_"+"w" + intlist2str(windows)+ "K"+intlist2str(Ks)+"Q"+intlist2str(Qs) + "_times.npy", np.array(allTimes))
    np.save(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" + str(nqueries)+"X"+str(nreferences)
            + "_X3_"+"w" + intlist2str(windows)+ "K"+intlist2str(Ks)+"Q"+intlist2str(Qs) + "_nboxes.npy", np.array(allnboxes))

    print("Data collection is done.")

def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2],machineRatios=[1,1]):
    datasets = []
    # with open(pathUCRResult+"allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    window = windows[0]
    rdtw = machineRatios[0]
    rother = machineRatios[1]
    t1dtw = loadt1dtw(pathUCRResult, maxdim, window)

#    datasets = ["ArticularyWordRecognition", "AtrialFibrillation"]

    ndatasets = len(datasets)

    # compute speedups
    x3setupLBtimes = np.load(
        pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/" + str(nqueries) + "X" + str(nreferences)
        + "_X3_w" + intlist2str(windows) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs)+"_times.npy")
    x3tLB = x3setupLBtimes[:,1]
    tCore = []
    skips = []
    ## -------------------
    NPairs = []
    if nqueries * nreferences == 0:
        actualNQNRs = np.loadtxt(pathUCRResult + '/usabledatasets_nq_nref.txt').reshape((-1, 2))
        for i in range(len(datasets)):
            actualNQ = actualNQNRs[i][0]
            actualNR = actualNQNRs[i][1]
            NPairs.append(actualNQ * actualNR)
    ## -------------------
    for dataset in datasets:
        for K in Ks:
            for Q in Qs:
                results = readResultFile(
                    pathUCRResult + dataset + '/d' + str(maxdim) + "/w" + str(windows[0]) + "/" + str(nqueries) + "X" + str(
                        nreferences) + "_X3_K" + str(K) + "Q"+str(Q)+"_results.txt")
                tCore.append(sum(results[:, 3]))
                skips.append(sum(results[:, 2]))
    tCore = np.array(tCore).reshape((ndatasets, -1))
    skips = np.array(skips).reshape((ndatasets, -1))

    tCorePlus = tCore + x3tLB.reshape((ndatasets,-1))
    tDTW = np.tile(t1dtw[0:ndatasets], (skips.shape[1], 1)).transpose() * ((skips - NPairs) * -1)
    tsum = rother * tCorePlus + rdtw * tDTW
    tsum_min = np.min(tsum, axis=1)
    setting_chosen = np.argmin(tsum,axis=1)
    skips_chosen = np.array( [skips[i,setting_chosen[i]] for i in range(skips.shape[0])] )
    overhead = rother* np.array([tCorePlus[i,setting_chosen[i]] for i in range(tCorePlus.shape[0])])
    speedups = (rdtw * t1dtw[0:ndatasets] * NPairs) / tsum_min
    overheadrate = overhead/(rdtw * t1dtw[0:ndatasets] * NPairs)

    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_skipschosen.npy', skips_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_settingchosen.npy', setting_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_overheadrate.npy', overheadrate)

    return 0


###############
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
    datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    Ks_g = [4, 6, 8]
    Qs_g = [2, 3, 4]
    windows_g = [20]
    dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g,nqueries_g,nreferences_g,windows_g,Ks_g,Qs_g)
    dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g,nqueries_g,nreferences_g,windows_g,Ks_g,Qs_g)

    print("End")