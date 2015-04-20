import Cache
import time
import datetime
from collections import Counter, defaultdict
import math
import nltk

def main():
	sFileTrain = 'test.e' #'hansards.36.2.e'
	tFileTrain = 'test.f' #'hansards.36.2.f'
	
	sFileTest = 'test.e'
	tFileTest = 'test.f'

	#sFile = 'corpus_1000.en'
	#tFile = 'corpus_1000.nl'

	#sFile = 'test.e'
	#tFile = 'test.f'

	print "Retrieving sentences and vocabularies..."
	sTest = getSentences('Data/'+sFileTest, 'Data/'+tFileTest)
	sTrain = getSentences('Data/'+sFileTrain, 'Data/'+tFileTrain)
	sentences = sTrain + sTest

	print '\tSentences:', str(len(sentences))
	global srcVoc
	global tarVoc
	srcVoc, tarVoc = getVocabularies(sentences, 'small_run-flip-3.e', 'small_run-flip-3.f')
	
	# store training vocabulary lengths
	global srcV
	srcV = len(srcVoc)
	print '\tsrcV:', str(srcV)
	global tarV
	tarV = len(tarVoc)
	print '\ttarV:', str(tarV)

	emTraining(sentences, sTest)

	#tsTable = Cache.cache('tsCache', dict())
	#outputViterbi(sentences, tsTable.cache, 'test.viterbi')

def getSentences(sFile, tFile):
    print sFile, tFile
    tk = nltk.tokenize.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    srcSens = []
    tarSens = []
    with open(sFile, 'rU') as sSnt:
        for line in sSnt:
            srcSens.append([word.lower() for word in tk.tokenize(line)])
            #if len(srcSens) is 7: break
    with open(tFile, 'rU') as tSnt:
        for line in tSnt:
            tarSens.append(['NULL']+[word.lower() for word in tk.tokenize(line)])
            #if len(tarSens) is 7: break
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
			#break
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
                for aj in range(len(tarSen)):
                    val = stTable[srcSen[j]][tarSen[aj]]
                    # verkeerde berekening?
                    senLL += math.log(val)
                    if val>maxVal:
                        maxVal = val
                        choice = aj
                # ommit NULL alignments
                if not choice is 0:
                	#outFile.write(str(i+1)+' '+str(j+1)+' '+str(choice)+'\n')
                    outFile.write('%04d %d %d\n'%(i+1, j+1, choice))
            likelihood += math.pow(math.e,senLL)
    print '\t\t\tLikelihood:', str(likelihood) 
    print '\t\tDuration:', getDuration(start, time.time())

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
    tarCounter = Counter(dict((t,1.0/tarV) for t in tarVoc))
    stTable = dict(zip(srcVoc,[tarCounter for s in srcVoc]))
    print 'stTable created ...'

    iteration = 0
    maxIter = 30
    while iteration<maxIter:
        print "Iteration " + str(iteration)
        start = time.time()
        counts = collectCounts(sentences, stTable)
        stTable = translationTable(counts)
        iteration+=1
        outputViterbi(sTest, stTable, 'Output/small_run-flip-3.viterbi.iter'+str(iteration))

	stTableCache = Cache.Cache('stTable.model1.iter'+str(maxIter), [])
	stTableCache.cache = stTable
    stTableCache.save()

def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()
