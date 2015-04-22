import Cache
import time
import datetime
from collections import Counter, defaultdict
import math
import nltk
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='Run IBM model 1 or model 2')
    parser.add_argument('-m', '--model', type=int, help='IBM model', required=True)
    parser.add_argument('-st', '--stInit', default='uniform', type=str, help='Method to initialize translation table', 
        required=False, choices=['uniform', 'random', 'model1'])
    parser.add_argument('-i', '--iter', default=15, type=int, help='Number of EM iterations', required=False)
    parser.add_argument('-t', '--test', default=False, type=bool, help='Test run (small set)', required=False)
    parser.add_argument('-sm', '--smooth', default=0, nargs=2, type=int, help='Smoothing parameters n and |V|. if |V| is 0 it is determined by the vocabulary present in the data', required=False)
    args = vars(parser.parse_args())

    #'none', 'smoothing', 'null-plus', 'heuristic', 'uniform', 'random', 'model1'

    global model
    model = args['model']
    
    global stInit
    if model is 1 and args['stInit'] == 'model1':
        print 'Initializing with model 1 only available for model 2. Defaulting to uniform.'
        stInit = 'uniform'
    else:
        stInit = args['stInit']

    global runType
    if args['test']:
        runType = 'test'
        sFileTrain = 'test.e'
        tFileTrain = 'test.f'
    else:
        runType = 'full_run'
        sFileTrain = 'hansards.36.2.e'
        tFileTrain = 'hansards.36.2.f'

    sFileTest = 'test.e'
    tFileTest = 'test.f'

    print "Retrieving sentences and vocabularies..."
    sTest = getSentences('Data/'+sFileTest, 'Data/'+tFileTest)
    sTrain = getSentences('Data/'+sFileTrain, 'Data/'+tFileTrain)
    sentences = sTrain + sTest

    print '\tSentences:', str(len(sentences))
    global srcVoc
    global tarVoc
    srcVoc, tarVoc = getVocabularies(sentences, runType+'.e', runType+'.f')

    # vocabularies do not differ between models
    runType += '.model'+str(args['model'])

    # store training vocabulary lengths
    global srcV
    srcV = len(srcVoc)
    print '\tsrcV:', str(srcV)
    global tarV
    tarV = len(tarVoc)
    print '\ttarV:', str(tarV)

    global smooth_n
    global smooth_v
    if args['smooth'] == None:
        smooth_n=0
        smooth_v=tarV
    else:
        smooth_n = args['smooth'][0]
        smooth_v = args['smooth'][1]

    global iterations
    iterations = args['iter']
    emTraining(sentences, sTest)

def getSentences(sFile, tFile):
    srcSens = []
    tarSens = []
    with open(sFile, 'rU') as sSnt:
        for line in sSnt:
            srcSens.append([word for word in line.split()])
            #if len(srcSens) is 2: break
    with open(tFile, 'rU') as tSnt:
        for line in tSnt:
            tarSens.append(['NULL']+[word for word in line.split()])
            #if len(tarSens) is 2: break
    return zip(srcSens, tarSens)

def getVocabularies(sentences, sFile, tFile):
	start = time.time()
	srcVoc = Cache.Cache(sFile+'.voc', [])
	tarVoc = Cache.Cache(tFile+'.voc', [])
	if not srcVoc.cache or not tarVoc.cache:
		for sSnt, tSnt in sentences:
			for s in sSnt:
				if not s in srcVoc.cache:
					srcVoc.cache.append(s)
			for t in tSnt:
				if not t in tarVoc.cache:
					tarVoc.cache.append(t)
		srcVoc.save()
		tarVoc.save()
	print 'Vocabularies obtained in', getDuration(start, time.time())
	return srcVoc.cache, tarVoc.cache

def outputViterbi(sentences, stTable, toFile):
    start = time.time()
    print "\tComputing Viterbi alignments ..."

    likelihood = 0
    with open(toFile,'w') as outFile:
         for i, (srcSen, tarSen) in enumerate(sentences):
            senLL = 0
            for j in range(len(srcSen)):
                maxVal = 0.0
                choice = 0
                alLL = 0
                for aj in range(len(tarSen)):
                    val = stTable[srcSen[j]][tarSen[aj]]
                    alLL += val
                    if val>maxVal:
                        maxVal = val
                        choice = aj
                # ommit NULL alignments
                if not choice is 0:
                	outFile.write('%04d %d %d\n'%(i+1, j+1, choice))
                senLL += math.log(alLL)
            likelihood += math.log(1e-5) - len(srcSen)*math.log(len(tarSen)+1) + senLL
    print '\t\t\tLikelihood:', str(likelihood) 
    print '\t\tDuration:', getDuration(start, time.time())
    return likelihood


def writeViterbiAligns(f, sents, stTable, alignP):
    start = time.time()
    print "\tComputing Viterbi alignments ..."

    with open(f,'w') as outFile:
         for i, (srcSen, tarSen) in enumerate(sents):
             l = len(tarSen)
             m = len(srcSen)
             for i in range(m):
                maxVal = 0.0
                choice = 0
                for j in range(l):
                    val = stTable[srcSen[i]][tarSen[j]]*alignP[(j+1,i+1,l,m)]
                    if val>maxVal:
                        maxVal = val
                        choice = j
                # ommit NULL alignments
                if not choice is 0:
                	outFile.write('%04d %d %d\n'%(i+1, i+1, choice))
                        
    print '\t\tDuration:', getDuration(start, time.time())

    
def logLikelihood(sentences, stTable, alignProbs):
    start = time.time()
    print "\tComputing likelihood ..."

    ll = 0

    for srcSen,tarSen in sentences:
        l = len(tarSen)
        m = len(srcSen)
        sll = 0
        for i in range(m):
            for j in range(l):
                sll += stTable[srcSen[i]][tarSen[j]]*alignProbs[(j+1,i+1,l,m)]
        ll += math.log(1e-5) - m*math.log(l+1) + math.log(sll)
    print '\t\t\tLikelihood:', str(ll) 
    print '\t\tDuration:', getDuration(start, time.time())
    return ll
'''
M-step
'''
def collectCounts(sentences, stTable,alignProbs={}):
    start = time.time()
    print "\tCollecting counts ..."

    counts = defaultdict(Counter)
    if model == 2:
        alignCj, alignC = initAligns(sentences)
    
    for srcSen, tarSen in sentences:
        # Compute normalization
        sTotals = Counter()
        saTotals = Counter()
        l = len(tarSen)
        m = len(srcSen)
        for i in range(m):
            for j in range(l):
                sTotals[tarSen[j]] += stTable[srcSen[i]][tarSen[j]]
                if model == 2:
                    saTotals[(i+1,l,m)] += alignProbs[(j+1,i+1,l,m)]*sTotals[tarSen[j]]
        # Collect counts
        for j in range(l):
            if sTotals[tarSen[j]]==0:
            	print tarSen[j] 
               	# sWord cannot be aligned to any word in tarSen
               	print 'sTotal is zero??!!'
            else:
                for i in range(m):
                    if model == 1:
                        value = stTable[srcSen[i]][tarSen[j]]/sTotals[tarSen[j]]
                    elif model == 2:
                        value = (stTable[srcSen[i]][tarSen[j]]*alignProbs[(j+1,i+1,l,m)])/saTotals[(i+1,l,m)]
                    if value > 0:
                        counts[srcSen[i]][tarSen[j]] += value
                        if model == 2:
                            alignCj[(j+1,i+1,l,m)] += value
                            alignC[(i+1,l,m)] += value
    print '\t\tDuration: ' + getDuration(start, time.time())
    if model == 1:
        return counts
    elif model == 2:
        return counts, alignCj, alignC

'''
E-step
'''
def translationTable(counts):
    start = time.time()
    print "\tRecomputing translation table ..."
    stTable = defaultdict(Counter)
    sTotals = Counter()

    for sWord, counter in counts.iteritems():
        sTotals[sWord] = sum(counter.values())
        for tWord, score in counter.iteritems():
            stTable[sWord][tWord] = (score + smooth_n)/(sTotals[sWord] + smooth_n * smooth_v)
    print '\t\tDuration: ' + getDuration(start, time.time())
    return stTable

def alignments(alignCj,alignC):
    start = time.time()
    print "\tRecomputing alignments ..."
    alignProbs = {}
    
    for (j,i,l,m) in alignCj:
        alignProbs[(j,i,l,m)] = alignCj[(j,i,l,m)]/alignC[(i,l,m)]
    print '\t\tDuration: ' + getDuration(start, time.time())
    return alignProbs

def initStTable():
    global stInit
    stTable = None
    start = time.time()
    print '\tInitializing stTable...'

    if stInit == 'model1':
        cache = 'stTable.'+runType+'.iter15'
        stCache = Cache.Cache(cache, [])
        if not stCache.cache:
            print 'Initialization cache', cache, 'unavailable. Defaulting to uniform.'
            stInit = 'uniform'
        else:
            stTable = stCache.cache

    if stInit == 'uniform':
        tarCounter = Counter(dict((t,1.0/tarV) for t in tarVoc))
        stTable = dict(zip(srcVoc,[tarCounter for s in srcVoc]))
    
    if stInit == 'random':
        tarCounter = []
        for s in srcVoc:
            values = {t:random.random() for t in tarVoc}
            totalValue = sum(values.values())
            tarCounter.append(Counter(dict((t,values[t]/totalValue) for t in tarVoc)))
        stTable = dict(zip(srcVoc,tarCounter))
        
    print '\tstTable created ...'
    print '\t\tDuration:', getDuration(start, time.time())
    return stTable

def initAligns(sentences):
    
    alignCj = {}
    alignC = {}
    for srcSen, tarSen in sentences:
        l = len(tarSen)
        m = len(srcSen)
        for i in range(m):
            alignC[(i+1,l,m)] = 0
            for j in range(l):
                alignCj[(j+1,i+1,l,m)] = 0
    return alignCj,alignC
        
def emTraining(sentences, sTest):
    print 'Beginning EM training...'
    globalStart=time.time()

    stTable = initStTable()
    if model == 2:
        alignCj, alignC = initAligns(sentences)
        unifAlignP = (float)(1)/len(alignCj) 
        alignProbs = {align:unifAlignP for align in alignCj}
        likelihoods = []
        
    likelihoodCache = Cache.Cache(runType+'.likelihood', [])
    i = 0
    while i<iterations:
        print "Iteration " + str(i)
        start = time.time()

        if model == 1:
            stCache = Cache.Cache('stTable.'+runType+'.iter'+str(i), [])
            if not stCache.cache:
                counts = collectCounts(sentences, stTable)
                stTable = translationTable(counts)
                stCache.cache = stTable
                if (i+1)%5 is 0:
                    stCache.save()
            else:
                stTable = stCache.cache
            viterbiFile = 'Output/'+runType+'.viterbi.iter'+str(i)
            likelihood = math.pow(math.e,(outputViterbi(sTest, stTable, viterbiFile)))
            likelihoodCache.cache.append(likelihood)
            likelihoodCache.save()
        
        elif model == 2:
            counts, alignCj, alignC = collectCounts(sentences, stTable, alignProbs)
            stTable = translationTable(counts)
            alignProbs = alignments(alignCj,alignC)
            ll = logLikelihood(sentences,stTable,alignProbs)
            likelihoods.append(ll)
            print '\t\tlikelihood M2: ', ll
        i+=1

    if model == 2:
        f = 'Output/'+runType+'.viterbi.final' 
        writeViterbiAligns(f, sTest,stTable, alignProbs)

    print "EMtraining finished after", iterations, "iterations in", getDuration(globalStart,time.time()),"."
    
def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()
