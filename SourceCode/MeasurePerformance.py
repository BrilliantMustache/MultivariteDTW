from SourceCode.Util import *
from SourceCode.X1 import DTWwnd
from SourceCode.Xz3 import DTWwbbox

# Main Entry Point
pathUCRResult = "../Results/UCR/"
datapath = "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
maxdim = 5
windowsize = 20
nqueries = 1
nreferences = 3
Ks_g = [4, 6, 8]
Qs_g = [2, 3, 4]

timesfile = '../Results/UCR/_AllDataSets/d'+ str(maxdim) + '/' + str(nqueries) + 'X' + \
             str(nreferences) + '_X0X1X3'+ 'K' + intlist2str(Ks_g) + 'Q' + intlist2str(Qs_g) +'_DTWTimes.npy'
t1dtwfile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+'/'+'Any_Anyw'+str(windowsize)+'_t1dtw.npy'
t1ndfile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+'/'+'Any_Anyw'+str(windowsize)+'_t1nd.npy'
t1bbfile = pathUCRResult+'_AllDataSets/d'+str(maxdim)+'/'+'Any_Anyw'+str(windowsize)+'_t1bb.npy'

#times = np.load(timesfile)

datasets = []
with open(pathUCRResult + "allDataSetsNames_no_EigenWorms.txt", 'r') as f:
    for line in f:
        datasets.append(line.strip())
f.close()
datasize = []
with open(pathUCRResult + "size_no_EigenWorms.txt", 'r') as f:
    for line in f:
        datasize.append(int(line.strip()))
f.close()

datasets = ["ArticularyWordRecognition", "AtrialFibrillation"]

alltimes=[]
t1nd=[]
t1bb=[]
for idxset, dataset in enumerate(datasets):
    print(dataset + " Start!")
    assert (datasize[idxset] >= nqueries + nreferences)
    stuff = loadUCRData_norm_xs(datapath, dataset, nqueries + nreferences)
    size = len(stuff)
    length = stuff[0].shape[0]
    dim = min(stuff[0].shape[1], maxdim)
    print("Size: " + str(size))
    print("Dim: " + str(dim))
    print("Length: " + str(length))
    samplequery = stuff[:nqueries]
    samplereference = stuff[nqueries:nreferences + nqueries]

    print(dataset + ":  " + str(nqueries) + " queries, " + str(nreferences) + " references." +
          " Total dtw: " + str(nqueries * nreferences))

    query = [q.values[:, :dim] for q in samplequery]
    reference = [r.values[:, :dim] for r in samplereference]

    templist = []
    totalpairs = len(query)*len(reference)
    # measure the times of DTW
    start = time.time()
    dtw = [DTW(s1, s2, windowsize) for s2 in reference for s1 in query]
    end = time.time()
    timedtw = (end - start)/totalpairs
    templist.append(timedtw)

    # measure the time of DTWwnd
    start = time.time()
    dtwwnd = [DTWwnd(s1, s2, windowsize) for s2 in reference for s1 in query]
    end = time.time()
    timedtwnd = (end - start)/totalpairs
    templist.append(timedtwnd)
    t1nd.append(max(0, timedtwnd - timedtw))

    # measure the time of DTWwbbox
    t1bb_1 = []
    for k in Ks_g:
        for q in Qs_g:
            start = time.time()
            dtwwbbox = [DTWwbbox(s1, s2, windowsize, k, q) for s2 in reference for s1 in query]
            end = time.time()
            timedtwwb = (end-start)/totalpairs
            templist.append(timedtwwb)
            t1bb_1.append(max(0, timedtwwb - timedtw))
    t1bb.append(t1bb_1)
    alltimes.append(templist)
                     #(timedtwnd-timedtw)/timedtw, (timedtwbbox-timedtw)/timedtw])

np.save(timesfile, np.array(alltimes))
np.save(t1dtwfile, np.array(alltimes)[:,0])
np.save(t1ndfile, np.array(t1nd))
np.save(t1bbfile, np.array(t1bb))

print('Done.')