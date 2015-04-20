import Cache
import time
import datetime
from collections import Counter, defaultdict
import math
import nltk

def main():
	eFileTrain = 'test.e'
	fFileTrain = 'test.f'
	
	eFileTest = 'test.e'
	fFileTest = 'test.f'

	#eFile = 'corpus_1000.en'
	#fFile = 'corpus_1000.nl'

	#eFile = 'test.e'
	#fFile = 'test.f'

	print "Retrieving sentences and vocabularies..."
	test = getSentences('Data/'+eFileTest, 'Data/'+fFileTest)
	train = getSentences('Data/'+eFileTrain, 'Data/'+fFileTrain)
	sentences = train + test

	print '\tSentences:', str(len(sentences))
	global eVoc
	global fVoc
	eVoc, fVoc = getVocabularies(sentences, 'small_run-flip-3.e', 'small_run-flip-3.f')
	
	# store training vocabulary lengths
	global eV
	eV = len(eVoc)
	print '\teV:', str(eV)
	global fV
	fV = len(fVoc)
	print '\tfV:', str(fV)

	emTraining(sentences, Test)

	#tsTable = Cache.cache('tsCache', dict())
	#outputViterbi(sentences, tsTable.cache, 'test.viterbi')

def emTraining(sentences, Test):
    print 'Beginning EM training...'
    eCounter = Counter(dict((t,1.0/eV) for e in eVoc))
    tfe = dict(zip(fVoc,[Counter for s in eVoc]))
    print 'tfe created ...'
	
    iteration = 0
    while iteration<30:
        print "Iteration " + str(iteration)
        start = time.time()
        counts = collectCounts(sentences, tfe)
        tfe = translationTable(counts)
        iteration+=1
        outputViterbi(Test, tfe, 'Output/small_run-flip-3.viterbi.iter'+str(iteration))

'''
M-step
'''
def collectCounts(sentences, tfe):
    start = time.time()
    print "\tCollecting counts ..."

    counts = defaultdict(Counter)

    for eSen, fSen in sentences:
        # Compute normalization
        fe = Counter()
        for e in eSen:
        	for f in fSen:
        		tf[f] += tfe[f][e]
        # Collect counts
        for f in fSen:
            if tf[f]== 0:
            	print f 
               	# sWord cannot be aligned to any word in tarSen
               	print 'sTotal is zero??!!'
            else:
                for e in eSen:
                    value = tfe[f][e]/tf[f]
                    if value > 0:
                        counts[e][f] += value
    print '\t\tDuration: ' + getDuration(start, time.time())
    return counts

'''
E-step
'''
def translationTable(counts):
    start = time.time()
    print "\tRecomputing translation table ..."
    tfe = defaultdict(Counter)
    tf = Counter()

    for e, counter in counts.iteritems():
        ce[e] = sum(counter.values())
        for f, score in counter.iteritems():
            tfe[f][e] = score/ce[e]
    print '\t\tDuration: ' + getDuration(start, time.time())
    return stTable


def outputViterbi(sentences, stTable, toFile):
    start = time.time()
    print "\tComputing Viterbi alignments ..."

    likelihood = 0
    with open(toFile,'w') as outFile:
         for i, (srcSen, tarSen) in enumerate(sentences):
            for j in range(len(srcSen)):
                maxVal = 0.0
                choice = 0
                for aj in range(len(tarSen)):
                    val = stTable[srcSen[j]][tarSen[aj]]
                    # verkeerde berekening?
                    likelihood += math.log(val)
                    if val>maxVal:
                        maxVal = val
                        choice = aj
                # ommit NULL alignments
                if not choice is 0:
                	outFile.write(str(i+1)+' '+str(j+1)+' '+str(choice)+'\n')
    print '\t\t\tLikelihood:', str(likelihood) 
    print '\t\tDuration:', getDuration(start, time.time())



def getSentences(sFile, tFile):
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

def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()
