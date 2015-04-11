import nltk
from math import log
from multiprocessing import Pool

class proj1(object):

    def __init__(self):
        self.enC = self.loadData('corpus_1000.en')
        self.nlC = self.loadData('corpus_1000.nl')

        # append 'NULL' to english sentences
        # and initialize t(f|e) 
        self.tfe = {}
        LMs = []
        for i in range(len(self.enC)):
            self.enC[i].append('NULL')
            for f in self.nlC[i]:
                l = len(self.enC[i])
                m = len(self.nlC[i])
                if (l,m) not in LMs:
                    LMs.append((l,m))
                for e in self.enC[i]:
                    self.tfe[(f,e)] = 1

        # normalize tfe to uniform distribution
        unifP = log( (float) (1)/len(self.tfe))
        for p in self.tfe:
            self.tfe[p] = unifP

        # set up the data structures 
        cjilms_all = [[(j+1,i+1,l,m) for j in range(m) for i in range(l) ] for (l,m) in LMs]
        self.cjilm = {}
        self.cilm = {}
        for cjilms in cjilms_all:
            for (j,i,l,m) in cjilms:
                self.cjilm[(j,i,l,m)] = 0
                self.cilm[(i,l,m)] = 0

        self.qjilm = self.cjilm
        self.IBM1(self.enC,self.nlC)
        
    # following Collinls lecture notes p.21 
    def IBM1(self, enC, nlC,T=5):

        it = 0
        while T > 0:
            print('iteration: ', it)
            it += 1
            
            self.cef = {(e,f):0 for (f,e) in self.tfe}
            self.ce = {e: 0 for (f,e) in self.tfe}
            self.cjilm = {c:0 for c in self.cjilm}
            self.cilm = {c:0 for c in self.cilm}
        
            print('maximizing...')
            # use 4 cpu's to count
            with Pool(4) as p:
                p.map(self.maximize,[(enC[:250],nlC[:250]),(enC[251:500],nlC[251:500]),(enC[501:750],nlC[501:750]),(enC[751:1000],nlC[751:1000])])
        
            print('estimating...')
            self.tfe = {(f,e):(self.cef[e,f]/self.ce[e]) for (f,e) in self.tfe}
            self.qjilm = {(j,i,l,m):(self.cjilm[(j,i,l,m)]/self.cilm[(i,l,m)]) for (j,i,l,m) in self.qjilm}

            T -= 1


    def maximize(self,sents):
        enC,nlC = sents
        
        for k in range(len(enC)):
            m = len(nlC[k])
            l = len(enC[k])
            for j in range(m):
                for i in range(l):
                    e = enC[k][i]
                    f = nlC[k][j]
                    delta = self.tfe[(f,e)] / sum( [ self.tfe[(x,y)] for (x,y) in self.tfe if x == f])
                    self.cef[(e,f)] += delta
                    self.ce[e] += delta
                    self.cjilm[(j+1,i+1,l,m)] += delta
                    self.cilm[(i+1,l,m)] += delta

        
    
    def loadData(self, fileloc):
        toker = nltk.tokenize.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
        st = nltk.stem.porter.PorterStemmer()
        
        sents = []
        with open(fileloc) as f:
            for s in f.readlines():
                sents.append([w.lower() for w in toker.tokenize(s)])
                
        return sents


if __name__=="__main__":
    proj1()
