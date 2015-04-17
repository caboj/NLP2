import nltk
from math import log, e, pow
from multiprocessing import Pool

class proj1(object):

    def __init__(self,model=2):
        enC = self.loadData('corpus_1000.en')
        nlC = self.loadData('corpus_1000.nl')

        # append 'NULL' to english sentences
        # and initialize t(f|e) 
        self.tfe = {}
        LMs = []
        for k in range(len(enC)):
            enC[k].append('NULL')
            for f in nlC[k]:
                l = len(enC[k])
                m = len(nlC[k])
                if (l,m) not in LMs:
                    LMs.append((l,m))
                for e in enC[k]:
                    self.tfe[(f,e)] = 1

        # normalize tfe to uniform distribution
        unifP = (float) (1)/len(self.tfe)
        for p in self.tfe:
            self.tfe[p] = unifP

        # set up the data structures 
        cjilms_all = [[(j+1,i+1,l,m) for j in range(l) for i in range(m) ] for (l,m) in LMs]
        self.cjilm = {}
        self.cilm = {}
        for cjilms in cjilms_all:
            for (j,i,l,m) in cjilms:
                self.cjilm[(j,i,l,m)] = 0
                self.cilm[(i,l,m)] = 0

        if model==1:
            self.qjilm = self.cjilm
        elif model == 2:
            unifQ = (float) (1)/len(self.cjilm)
            self.qjilm = {q:unifQ for q in self.cjilm}
        ltqs = self.IBM(enC,nlC,1)

        print(ltqs)
    # following Collinls lecture notes p.21 
    def IBM(self, enC, nlC,model,T=5):

        ltqs = []
        it = 0
        while T > 0:
            print('iteration: ', it)
            it += 1
            
            self.cef = {(e,f):0 for (f,e) in self.tfe}
            self.ce = {e: 0 for (f,e) in self.tfe}
            self.cjilm = {c:0 for c in self.cjilm}
            self.cilm = {c:0 for c in self.cilm}
        
            print('maximizing...')
            self.maximize((enC,nlC),1)
            # use 4 cpu's to count
            #with Pool(4) as p:
            #    p.map(self.maximize,[(enC[:250],nlC[:250]),(enC[251:500],nlC[251:500]),(enC[501:750],nlC[501:750]),(enC[751:1000],nlC[751:1000])])
        
            print('estimating...')
            self.tfe = {(f,e):(self.cef[e,f]/self.ce[e]) for (f,e) in self.tfe}
            self.qjilm = {(j,i,l,m):(self.cjilm[(j,i,l,m)]/self.cilm[(i,l,m)]) for (j,i,l,m) in self.qjilm}

            # determine likelihood
            ltqs.append((T,self.ltq(enC,nlC)))
            T -= 1

        return ltqs
    def maximize(self,sents,model):
        enC,nlC = sents

        # precompute denominators of delta
        tf = {}
        for (f,e) in self.tfe:
            if f in tf:
                tf[f] += self.tfe[(f,e)]
            elif f not in tf:
                tf[f] = self.tfe[(f,e)]

        qilm = {}
        for (j,i,l,m) in self.qjilm:
            if (i,l,m) in qilm:
                qilm[(i,l,m)] += self.qjilm[(j,i,l,m)]
            elif (i,l,m) not in qilm:
                qilm[(i,l,m)] = self.qjilm[(j,i,l,m)]

        # compute new counts to maximize estimation
        for k in range(len(enC)):
            m = len(nlC[k])
            l = len(enC[k])
            for i in range(m):
                for j in range(l):
                    e = enC[k][j]
                    f = nlC[k][i]
                    if model == 1:
                        delta = self.tfe[(f,e)] / tf[f]
                    elif model == 2:
                        delta = self.tfe[(f,e)]*self.qjilm[(j+1,i+1,l,m)] / (tf[f]+qilm[i,l,m])
                    self.cef[(e,f)] += delta
                    self.ce[e] += delta
                    self.cjilm[(j+1,i+1,l,m)] += delta
                    self.cilm[(i+1,l,m)] += delta

    
    def ltq(self, enC, nlC):
        pf = 0
        for k in range(len(enC)):
            pf_k = 0
            l = len(enC[k])
            m = len(nlC[k])
            for i in range(len(nlC[k])):
                pfa_k = 0
                for j in range(len(enC[k])):                
                    if ((j+1, i+1,l,m) in self.qjilm and (nlC[k][i],enC[k][j]) in self.tfe):
                        pfa_k += self.qjilm[j+1,i+1,l,m]*self.tfe[(nlC[k][i],enC[k][j])]
                pf_k += log(pfa_k)
            pf+=pow(e,pf_k)
        return pf
            
                
    
    def loadData(self, fileloc):
        toker = nltk.tokenize.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
        #st = nltk.stem.porter.PorterStemmer()
        
        sents = []
        with open(fileloc) as f:
            for s in f.readlines():
                sents.append([w.lower() for w in toker.tokenize(s)])
                
        return sents


if __name__=="__main__":
    proj1()
