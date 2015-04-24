from collections import Counter, defaultdict
import numpy as np
import Cache
import time
import datetime
import math
import argparse
import random

def main():
	sFileTrain = 'hansards.36.2.e'
	tFileTrain = 'hansards.36.2.f'

	sFileTest = 'test.e'
	tFileTest = 'test.f'

	print "Retrieving sentences and vocabularies..."
	sTest = getSentences('Data/'+sFileTest, 'Data/'+tFileTest)
	sTrain = getSentences('Data/'+sFileTrain, 'Data/'+tFileTrain)
	sentences = sTrain + sTest

	runType = 'full_run'
	print '\tSentences:', str(len(sentences))
	global srcVoc
	global tarVoc
	srcVoc, tarVoc = getVocabularies(sentences, runType+'.e', runType+'.f')

	global srcV
	srcV = len(srcVoc)
	print '\tsrcV:', str(srcV)
	global tarV
	tarV = len(tarVoc)
	print '\ttarV:', str(tarV)

	initStTable()

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

def initStTable():
    global stInit
    stTable = None
    start = time.time()
    print '\tInitializing stTable...'

    stCache = Cache.Cache('stTable.random.full_run.v1', {})
    tOnes = np.ones(tarV)
    for s in xrange(srcV):
        if s%10 is 0:
            print '\t\t'+str(s)
        values = zip(tarVoc, np.random.dirichlet(tOnes,size=1)[0])
        #stTable[srcVoc[s]] = Counter(dict(zip(tarVoc,values)))
        stCache.cache[srcVoc[s]] = Counter(dict(values))
    stCache.save()

    print '\tRandom stTable created ...'
    print '\t\tDuration:', getDuration(start, time.time())

def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()
