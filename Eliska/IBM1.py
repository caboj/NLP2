import Cache
import time
import datetime
from collections import Counter, defaultdict

def main():
	#sFile = 'hansards.36.2.e'
	#tFile = 'hansards.36.2.f'
	
	sFile = 'corpus_1000.en'
	tFile = 'corpus_1000.nl'

	#sFile = 'test.e'
	#tFile = 'test.f'

	print "Retrieving sentences and vocabularies..."
	sentences = getSentences('Data/'+sFile, 'Data/'+tFile)
	print 'sentences:', str(len(sentences))
	global srcVoc
	global tarVoc
	srcVoc, tarVoc = getVocabularies(sentences, sFile, tFile)
	# store vocabulary lengths
	global srcV
	srcV = len(srcVoc)
	print 'srcV:', str(srcV)
	global tarV
	tarV = len(tarVoc)
	print 'tarV:', str(tarV)

	emTraining(sentences)

	#tsTable = Cache.cache('tsCache', dict())
	#outputViterbi(sentences, tsTable.cache, 'test.viterbi')

def getSentences(sFile, tFile):
    srcSens = []
    tarSens = []
    with open(sFile, 'rU') as sSnt:
        for line in sSnt:
            srcSens.append([word for word in line.split()]+['NULL'])
            #if len(srcSens) is 7: break
    with open(tFile, 'rU') as tSnt:
        for line in tSnt:
            tarSens.append([word for word in line.split()])
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

def outputViterbi(sentences, tsTable, toFile):
    start = time.time()
    print "\tComputing Viterbi alignments ..."

    with open(toFile,'w') as outFile:
         for i, (srcSen, tarSen) in enumerate(sentences):
            for j in range(len(tarSen)):
                maxVal = 0.0
                choice = 0
                for aj in range(len(srcSen)):
                    val = tsTable[tarSen[j]][srcSen[aj]]
                    if val>maxVal:
                        maxVal = val
                        choice = aj
                # ommit NULL alignments
                if (choice < len(srcSen)-1):
                	outFile.write(str(i+1)+' '+str(j+1)+' '+str(choice+1)+'\n')
    print '\t\tDuration: ' + getDuration(start, time.time())

'''
M-step
'''
def collectCounts(sentences, tsTable):
    start = time.time()
    print "\tCollecting counts ..."

    counts = defaultdict(Counter)

    for srcSen, tarSen in sentences:
        # Compute normalization
        tTotals = Counter()
        for t in tarSen:
        	for s in srcSen:
        		tTotals[s] += tsTable[t][s]
        # Collect counts
        for sWord in srcSen:
            if tTotals[sWord]== 0:
            	print sWord 
               	# sWord cannot be aligned to any word in tarSen
               	print 'tTotal is zero??!!'
            else:
                for tWord in tarSen:
                    value = tsTable[tWord][sWord]/tTotals[sWord]
                    if value > 0:
                        counts[tWord][sWord] += value
    print '\t\tDuration: ' + getDuration(start, time.time())
    return counts

'''
E-step
'''
def translationTable(counts):
    start = time.time()
    print "\tRecomputing translation table ..."
    stTable = defaultdict(Counter)
    tsTable = defaultdict(Counter)


    #sTotals = [sum([counts[t][s] for t in counts.iterkeys()]) for s in srcVoc]
    tTotals = Counter()

    for tWord, counter in counts.iteritems():
        for sWord, score in counter.iteritems():
        	tTotals[tWord] = sum(counter.values())
        	tsTable[tWord][sWord] = score/tTotals[tWord]
    print '\t\tDuration: ' + getDuration(start, time.time())
    return tsTable

def emTraining(sentences):
	print 'Beginning EM training...'
	srcCounter = Counter(dict((s,1.0/srcV) for s in srcVoc))
	tsTable = dict(zip(tarVoc,[srcCounter for t in tarVoc]))
	print 'tsTable created ...'
	#print 'TSTABLE'
	#print tsTable

	iteration = 0
	while iteration<15:
		print "Iteration " + str(iteration)
		start = time.time()
		counts = collectCounts(sentences, tsTable)
		#print counts
		tsTable = translationTable(counts)
		#$print 'TSTABLE'
		#print tsTable
		iteration+=1

	#tsCache = Cache.cache('tsCache', tsTable)
	#tsCache.save()
	outputViterbi(sentences, tsTable, 'test.viterbi')

def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()