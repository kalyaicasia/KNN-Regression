import numpy as np
class knnreg ():
    def __init__(self, distfcn = "euclidean", normfcn = "on") :
        if distfcn.lower() == "euclidean": self.q = 2
        else: self.q = 1
            
        self.normfn = normfcn.lower()
        
    def colcheck(self,ma):                  #fix the data input if the input is only 1 collumn of data
        try: 
            len(ma[0])
        except TypeError: 
            ma = [[x] for x in ma]
        return ma
    
    def rowcheck(self, ma):            #fix the data input if the input is only 1 row of data
        try: 
            len(ma[0])
        except TypeError: 
            ma = [ma]
        return ma
        
    def normalize(self, da, atrc):          #counts the maxmin values for each atr, normalizes all atribute values
        self.max[atrc] = max(da)
        self.min[atrc] = min(da)
        self.den[atrc] = self.max[atrc] - self.min[atrc]

        return [((x-self.min[atrc])/(self.den[atrc])) for x in da]
    
    def calcdist(self, nd, ed):                  #calculates euclidean distance
        sigma = sigma = [(abs(nd[ed.index(x)]-x))**self.q for x in ed]
        distance = (sum(sigma))**(1/self.q)
        
        return distance
    
    def fit (self, dec, tar):              #initializes variables, obtains the needed values from data
        tar = self.colcheck(tar)
        
        self.tar = tar
        self.samples = len(tar)
        self.atr = len(dec)

        #normalize values to match each data to scale if the function is ON
        if self.normfn == "on":
            #transform data matrix so that the first index refers to each attributes              
            self.alt = np.array(dec)  
            self.dec = self.alt.T.tolist()                 #self.dec - index refers to atributes
            
            self.min = [0]*self.atr
            self.max = [0]*self.atr
            self.den = [0]*self.atr
            self.dec = [self.normalize(x,self.dec.index(x)) for x in self.dec] 
            
            self.alt = np.array(self.dec).T.tolist()       #self.alt - index refers to samples
            
        else: self.alt = dec
        
        print("Model Fitted")
        
    def predicteach(self, k, nd):
        if self.normfn == "on":
            i=0
            for x in nd:
                nd[i] = (nd[i]-self.min[i])/(self.den[i])
                i=i+1
        
        #calculate distances
        dist = [self.calcdist(nd,x) for x in self.alt]

        #find the k closest targets
        kc = k
        dsum = 0
        exd = dist
        while kc > 0 :
            loc = dist.index(min(exd))        #find index of the closest datas
            dsum = dsum + self.tar[loc][0]     #total prediced from obtained index
            exd[loc]=max(exd)                 #remove the current min value 
            kc = kc-1                        #counts how many more predicitions are needed            
            
        #calculate predicted value 
        pv = dsum / k 
        return pv
    
    def predict(self, k, nd):
        nd = self.rowcheck(nd)
        
        predv = [0]*len(nd)
        i = 0
        while i<len(nd):
            predv[i] = [self.predicteach(k,nd[i])]
            i=i+1
        
        return predv
    
    def check(self, pv, nt):
        pv = self.colcheck(pv)
        nt = self.colcheck(nt)
        
        count = [0]*len(pv)
        i = 0
        for x in count:
            count [i] = abs(pv[i][0] - nt[i][0])/nt[i][0]
            i = i+1
        error = (sum(count)/len(pv))*100
        acc = 100 - error
        
        return error, acc