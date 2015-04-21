import Cache
import time
import datetime
from collections import Counter, defaultdict
import math
import nltk
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run IBM model 1 or model 2')
    parser.add_argument('-m', '--model', type=int, help='IBM model', required=True)
    parser.add_argument('-i', '--iter', default=15, type=int, help='Number of EM iterations', required=False)
    parser.add_argument('-t', '--test', default=False, type=bool, help='Test run (small set)', required=False)
    args = vars(parser.parse_args())

    global model
    model = args['model']

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

    global iterations
    iterations = args['iter']
    emTraining(sentences, sTest)

def getSentences(sFile, tFile):
    srcSens = []
    tarSens = []
    with open(sFile, 'rU') as sSnt:
        for line in sSnt:
            srcSens.append([word for word in line.split()])
            #if len(srcSens) is 70: break
    with open(tFile, 'rU') as tSnt:
        for line in tSnt:
            tarSens.append(['NULL']+[word for word in line.split()])
             #if len(tarSens) is 70: break
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

'''
M-step
'''
def collectCounts(sentences, stTable):
    start = time.time()
    print "\tCollecting counts ..."

    counts = defaultdict(Counter)

    for srcSen, tarSen in sentences:
        # Compute normalization
        sTotals = Counter()
        for s in srcSen:
        	for t in tarSen:
        		sTotals[t] += stTable[s][t]
        # Collect counts
        for tWord in tarSen:
            if sTotals[tWord]==0:
            	print tWord 
               	# sWord cannot be aligned to any word in tarSen
               	print 'sTotal is zero??!!'
            else:
                for sWord in srcSen:
                    value = stTable[sWord][tWord]/sTotals[tWord]
                    if value > 0:
                        counts[sWord][tWord] += value
    print '\t\tDuration: ' + getDuration(start, time.time())
    return counts

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
            stTable[sWord][tWord] = score/sTotals[sWord]
    print '\t\tDuration: ' + getDuration(start, time.time())
    return stTable

def emTraining(sentences, sTest):
    print 'Beginning EM training...'
    globalStart=time.time()

    tarCounter = Counter(dict((t,1.0/tarV) for t in tarVoc))
    stTable = dict(zip(srcVoc,[tarCounter for s in srcVoc]))
    print 'stTable created ...'
	
    likelihoodCache = Cache.Cache(runType+'.likelihood', [])
    i = 0
    while i<iterations:
        print "Iteration " + str(i)
        start = time.time()
        
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
        likelihood = outputViterbi(sTest, stTable, viterbiFile)
        likelihoodCache.cache.append(likelihood)
        likelihoodCache.save()
        i+=1

    print "EMtraining finished after", iterations, "iterations in", getDuration(globalStart,time.time()),"."
    
def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()