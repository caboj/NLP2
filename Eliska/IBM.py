from collections import Counter, defaultdict
from scipy.special import digamma
import numpy as np
import Cache
import time
import datetime
import math
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='Run IBM model 1 or model 2')
    parser.add_argument('-m', '--model', type=int, help='IBM model', required=True, choices=[1, 2])
    parser.add_argument('-st', '--stInit', default='uniform', type=str, help='Method to initialize translation table', 
        required=False, choices=['uniform', 'random', 'model1', 'heuristic'])
    parser.add_argument('-i', '--iter', default=15, type=int, help='Number of EM iterations', required=False)
    parser.add_argument('-t', '--test', default=False, type=bool, help='Test run (small set)', required=False)
    parser.add_argument('-sm', '--smooth', default=None, nargs=2, type=int, required=False,
        help='Smoothing parameters n and |V|. If |V| is 0 it is determined by the vocabulary present in the data')
    parser.add_argument('-a', '--alpha', default=None, type=float, required=False, 
        help='Alpha parameter for Variational Bayes.')
    parser.add_argument('-nulls', '--nullWords', default=1, type=int, required=False, 
        help='Number of null words added to target sentence')
    args = vars(parser.parse_args())

    #'none', 'smoothing', 'null-plus', 'heuristic', 'uniform', 'random', 'model1'

    global model
    model = args['model']
    global test
    test = args['test']

    global stInit
    if model is 1 and args['stInit'] == 'model1':
        print 'Initializing with model 1 only available for model 2. Defaulting to uniform.'
        stInit = 'uniform'
    else:
        stInit = args['stInit']

    global nullN
    nullN = args['nullWords']

    global runType
    if args['test']:
        runType = 'test'
        sFileTrain = 'test.e'
        tFileTrain = 'test.f'
    else:
        runType = '10000'
        sFileTrain = 'hansards.36.2.e.10000'
        tFileTrain = 'hansards.36.2.f.10000'

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

    # vocabularies only differ between test runs and full runs
    runType += '.model'+str(args['model'])+'.'+args['stInit']
    if args['stInit'] == 'random':
        runType += '1'

    # store training vocabulary lengths
    global srcV
    srcV = len(srcVoc)
    print '\tsrcV:', str(srcV)
    global tarV
    tarV = len(tarVoc)
    print '\ttarV:', str(tarV)

    global smooth
    if not args['smooth']:
        smooth = {'n':0, 'v':tarV}
    else:
        smooth = {'n':args['smooth'][0], 'v':args['smooth'][1]}
        runType += '.n'+str(smooth['n'])+'.v'+str(smooth['v'])

    global alpha
    if not args['alpha'] == None:
        alpha = args['alpha']
        runType += '.alpha'+str(alpha)
        print runType
    else:
        alpha = None

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

def outputViterbi(sentences, stTable, toFile, alignP=None):
    start = time.time()
    print "\tComputing Viterbi alignments ..."

    with open(toFile,'w') as outFile:
         for i, (srcSen, tarSen) in enumerate(sentences):
            m = len(srcSen)
            l = len(tarSen)
            for j in xrange(m):
                maxVal = 0.0
                choice = 0
                for aj in xrange(l):
                    if model is 1:
                        val = stTable[srcSen[j]][tarSen[aj]]
                    else:
                        val = stTable[srcSen[j]][tarSen[aj]]*alignP[(aj+1,j+1,l,m)]
                    if val>maxVal:
                        maxVal = val
                        choice = aj
                # ommit NULL alignments
                if not choice is 0:
                    outFile.write('%04d %d %d\n'%(i+1, j+1, choice))
    print '\t\tDuration:', getDuration(start, time.time())
    
def logLikelihood(sentences, stTable, epsilon, alignProbs=None):
    print "\tComputing log likelihood ..."
    start = time.time()
    ll = 0
    for srcSen,tarSen in sentences:
        l = len(tarSen)
        m = len(srcSen)
        senLL = 0
        for j in xrange(m):
            alignLL = 1
            for aj in xrange(l):
                if model is 1:
                    senLL += stTable[srcSen[j]][tarSen[aj]]
                else:
                    alignLL *= stTable[srcSen[j]][tarSen[aj]]*alignProbs[(aj+1,j+1,l,m)]
            senLL += alignLL
        if model is 1:
            ll += math.log(epsilon) - m*math.log(l+1) + math.log(senLL)
        else:
            if senLL != 0.0:
                ll += math.log(epsilon) + math.log(senLL)
    print '\t\t\tLog likelihood:', str(math.pow(math.e,ll)) 
    print '\t\tDuration:', getDuration(start, time.time())
    return ll

'''
M-step
'''
def collectCounts(sentences, stTable, alignProbs=None):
    start = time.time()
    print "\tCollecting counts ..."

    counts = defaultdict(Counter)
    if model is 2:
        alignCj, alignC = initAligns(sentences)
    
    for srcSen, tarSen in sentences:
        # Compute normalization
        sTotals = Counter()
        saTotals = Counter()
        l = len(tarSen)
        m = len(srcSen)
        for j in xrange(m):
            for aj in xrange(l):
                if model is 1:
                    sTotals[tarSen[aj]] += stTable[srcSen[j]][tarSen[aj]]
                else:
                    #saTotals[(j+1,l,m)] += alignProbs[(aj+1,j+1,l,m)]*sTotals[tarSen[aj]]
                    saTotals[(j+1,l,m)] += alignProbs[(aj+1,j+1,l,m)]*stTable[srcSen[j]][tarSen[aj]]
        # Collect counts
        for aj in xrange(l):
            if model is 1 and sTotals[tarSen[aj]] is 0:
                print tarSen[aj] 
                # sWord cannot be aligned to any word in tarSen
                print 'sTotal is zero??!!'
            else:
                for j in xrange(m):
                    if model is 1:
                        value = stTable[srcSen[j]][tarSen[aj]]/sTotals[tarSen[aj]]
                    else:
                        value = (stTable[srcSen[j]][tarSen[aj]]*alignProbs[(aj+1,j+1,l,m)])/saTotals[(j+1,l,m)]
                    if value > 0:
                        counts[srcSen[j]][tarSen[aj]] += value
                        if model is 2:
                            alignCj[(aj+1,j+1,l,m)] += value
                            alignC[(j+1,l,m)] += value
    print '\t\tDuration: ' + getDuration(start, time.time())
    
    if model is 1:
        return counts
    elif model is 2:
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
            if not alpha is None:
                stTable[sWord][tWord] = nullN * math.pow(math.e,digamma(score+alpha))/math.pow(math.e,digamma(sTotals[sWord]+alpha))
            else:
                stTable[sWord][tWord] = nullN * (score+smooth['n'])/(sTotals[sWord]+smooth['n']*smooth['v'])
    print '\t\tDuration: ' + getDuration(start, time.time())
    return stTable

def alignments(alignCj, alignC):
    start = time.time()
    print "\tRecomputing alignments ..."
    alignProbs = {(aj,j,l,m):alignCj[(aj,j,l,m)]/alignC[(j,l,m)] for (aj,j,l,m) in alignCj}
    print '\t\tDuration: ' + getDuration(start, time.time())
    return alignProbs

def initStTable(sentences):
    global stInit
    stTable = None
    start = time.time()
    print '\tInitializing stTable...'

    if stInit == 'model1':
        if test:
            cache = 'stTable.test.model1.uniform.iter14'
        else:
            cache = 'stTable.10000.model1.uniform.iter14'
        stCache = Cache.Cache(cache, [], True)
        if not stCache.cache:
            print 'Initialization cache', cache, 'unavailable. Defaulting to uniform.'
            stInit = 'uniform'
        else:
            stTable = stCache.cache

    if stInit == 'uniform':
        tarCounter = Counter(dict((t,1.0/tarV) for t in tarVoc))
        stTable = dict(zip(srcVoc,[tarCounter for s in srcVoc]))
    
    if stInit == 'random':
        values = zip(tarVoc, np.random.dirichlet(np.ones(tarV),size=1)[0])
        tarCounter = Counter(dict(values))
        stTable = dict(zip(srcVoc,[tarCounter for s in srcVoc]))

    if stInit == 'heuristic':
        stCounts = {s:{t:0.0 for t in tarVoc if t != 'NULL'} for s in srcVoc}
        sTotals = {s:0.0 for s in srcVoc}
        tTotals = {t:0.0 for t in tarVoc if t != 'NULL'}
        sFreq = {s:0.0 for s in srcVoc}
        tFreq = {t:0.0 for t in tarVoc}
        # Count number of sentences any s or t appear in and total appearances of each word.
        for (src, tar) in sentences:
            seen = {}
            for t in tar[1:]:                            
                tFreq[t] += 1
                if t not in seen:
                    tTotals[t] += 1
                    seen[t] = True
            seen = {}
            for s in src:
                sFreq[s] += 1
                tSeen = {}
                if s not in seen:
                    sTotals[s] += 1
                    seen[s] = True
                for t in tar[1:]:
                    if t not in t_seen:
                        stCounts[s][t] += 1
                        tSeen[t] = True
        stTable = {s:{t:0.0 for t in tarVoc if t != 'NULL'} for s in srcVoc}
        # Calculate LLR
        for s in srcVoc:
            for t in t_totals.keys():
                stCount = stCounts[s][t]
                
                if stCount / len(sentences) > (sTotals[s] * tTotals[t]) / (len(sentences)**2):                    
                    stTable[s][t] =  st_count * math.log((stCount / sTotals[s]) / (tTotals[t] / len(sentences))) # s and t
                    try:
                        stTable[s][t] += (sTotals[s] - stCount) * math.log(((sTotals[s] - stCount) / sTotals[s]) / ((len(sentences)- tTotals[t]) /len(sentences))) # s and not t
                    except ValueError:
                        continue
                    try: 
                        stTable[s][t] += (tTotals[t] - stCount) * math.log(((tTotals[t] - stCount) / (len(sentences) - sTotals[s])) / (tTotals[t] / len(sentences))) # t and not s
                    except ValueError:
                        continue
                    try: 
                        stTable[s][t] += (len(sentences) - sTotals[s] - tTotals[t] + stCount) * \
                            math.log(((len(sentences) - sTotals[s] - tTotals[t] + stCount) / (len(sentences) - sTotals[s])) / ((len(sentences)- tTotals[t])/len(sentences)))# not s and not t
                    except ValueError:
                        continue
                else: #Negative correlation
                    stTable[s][t] = 0.0
        #Find max marginal value for s for normalization
        maxVal = 0.0
        for cond_t in stTable.values():
            if sum(condT.values()) > maxVal:
                maxVal = sum(condT.values())
        
        # Normalize
        for condT in stTable.values():
            for t in condT.keys():
                condT[t] = condT[t] / maxVal

        sTotalSum = sum([len(srcSen) for (srcSen, tarSen) in sentences])

        for s in stTable.keys():
            stTable[s]['NULL'] = nullN * (sTotals[s] / sTotalSum)
            stTable[s] = Counter(stTable[s])

    print '\tstTable created ...'
    print '\t\tDuration:', getDuration(start, time.time())
    return stTable

def initAligns(sentences):
    print '\tinitializing alignment counts (setting to zero)...'
    start = time.time()
    alignCj = {}
    alignC = {}
    for srcSen, tarSen in sentences:
        l = len(tarSen)
        m = len(srcSen)
        for j in xrange(m):
            alignC[(j+1,l,m)] = 0.0
            for aj in xrange(l):
                alignCj[(aj+1,j+1,l,m)] = 0.0
    print '\t\t\tDuration:', getDuration(start, time.time())
    return alignCj, alignC
        
def estimateEpsilon(sentences):
    # estimate fixed epsilon
    pl = defaultdict(Counter)
    for srcSen, tarSen in sentences:
        pl[len(srcSen)][len(tarSen)]+=1
       
    acc = 0
    for k in pl:
        acc += sum(pl[k])
    
    epsilon = 1.0/acc
    print '\tepsilon: ',epsilon
    return epsilon

def emTraining(sentences, sTest):
    print 'Beginning EM training...'
    globalStart=time.time()
    stTable = initStTable(sentences)
    epsilon = estimateEpsilon(sentences)

    if model is 2:
        alignCj, alignC = initAligns(sentences)
        unifAlignP = 1.0/len(alignCj) 
        #alignProbs = {align:unifAlignP for align in alignCj}
        alignProbs = dict(zip(alignCj, [unifAlignP]*len(alignCj)))
    else:
        alignProbs = None

    likelihoodCache = Cache.Cache(runType+'.likelihood', [])
    i = 0
    while i<iterations:
        print "Iteration " + str(i)
        start = time.time()

        stCache = Cache.Cache('stTable.'+runType+'.iter'+str(i), [], True)
        if not stCache.cache:
            if model is 1:
                counts = collectCounts(sentences, stTable)
                stTable = translationTable(counts)
            else:
                counts, alignCj, alignC = collectCounts(sentences, stTable, alignProbs)
                stTable = translationTable(counts)
                alignProbs = alignments(alignCj, alignC)
            stCache.cache = stTable
            if i is 0 or (i+1)%5 is 0:
                stCache.save()
        #else:
        #    stTable = stCache.cache
        viterbiFile = 'Output/'+runType+'.viterbi.iter'+str(i)
        outputViterbi(sTest, stTable, viterbiFile, alignProbs)
        likelihood = logLikelihood(sentences, stTable, epsilon, alignProbs)
        likelihoodCache.cache.append(likelihood)
        likelihoodCache.save()
        i+=1

    print "EMtraining finished after", iterations, "iterations in", getDuration(globalStart,time.time()),"."
    
def getDuration(start, stop):
    return str(datetime.timedelta(seconds=(stop-start)))

if __name__ == '__main__':
    main()
