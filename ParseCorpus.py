# coding: gbk
# basic process of corpus ...

import sys
sys.path.append('jieba')
import jieba
from operator import add
from pyspark import SparkContext
import math
import cPickle

MIN_NUM , MAX_RATE = 4, 0.1

def IDFmap(line):
    res = [ (x, 1) for x in jieba.cut(line.strip(), cut_all = True)]
    return list(set(res))

def process_dic(df, min, max, NUM_DOC):
    res = {}
    i = 0
    for line in df:
        if line[1] < min or line[1] > max * NUM_DOC:
            continue
        res[line[0]] = (i, math.log( NUM_DOC * 1.0  / (line[1] + 1.0) ) )
        i = i+1
    return res
        
def CorpusMap(line, dic):
    terms = jieba.cut(line.strip(), cut_all = True)
    terms = [x for x in terms]
    #print terms
    #print dic
    N = len(terms)
    res = {}
    for ele in terms:
        if ele in dic:
            (id, df) = dic[ele]
            if id in res:
                res[id][0] = res[id][0] + 1
            else:
                res[id] = [1,df]
    #print res
    res = [(x, (res[x][0] * 1.0 / N) * res[x][1] ) for x in res]
    res = " ".join( ["{0}:{1}".format(x[0],x[1]) for x in res] )
    return res
    
if __name__=="__main__":
    sc = SparkContext(appName="ParseCorpus")
    text = sc.textFile(sys.argv[1]).repartition(300).cache()
    NUM_DOC = text.count()
    print "total num of Doc: ", NUM_DOC
    df = text.flatMap(IDFmap).reduceByKey(add).collect()
    dic = process_dic(df, MIN_NUM , MAX_RATE, NUM_DOC)
    cPickle.dump(dic, open("dictionary.dic", 'w'), 2)
    #print CorpusMap("我爱北京天安门,我爱北京天安门", dic)
    corpus = text.map(lambda line : CorpusMap(line, dic))
    corpus.saveAsTextFile(sys.argv[2])
    

    
