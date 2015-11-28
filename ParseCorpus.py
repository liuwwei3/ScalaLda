# coding: gbk
# basic process of corpus ...

import sys
sys.path.append('jieba')
import jieba
from operator import add
from pyspark import SparkContext
import math
import cPickle

def IDFmap(line):
    return [ (x, 1) for x in jieba.cut(line.strip())]

def process_dic(df, min, max):
    res = {}
    i = 0
    for line in df:
        if line[1] < min or line[1] > max * NUM_DOC:
            continue
        res[line[0]] = (i, line[1])
        i = i+1
    return res
        
def CorpusMap(line, dic, NUM_DOC):
    terms = jieba.cut(line.strip())
    terms = [x for x in terms]
    #print terms
    #print dic
    N = len(terms)
    res = {}
    for ele in terms:
        if ele in dic:
            (id, df) = dic[ele]
            if id in res:
                res[id] = res[id][0] + 1
            else:
                res[id] = [1,df]
    #print res
    res = [(x, (res[x][0] * 1.0 / N) * math.log( NUM_DOC * 1.0  /(1.0 + res[x][1]) )) for x in res]
    res = " ".join( ["{0}:{1}".format(x[0],x[1]) for x in res] )
    return res
    
if __name__=="__main__":
    sc = SparkContext(appName="ParseCorpus")
    text = sc.textFile(sys.argv[1]).cache()
    NUM_DOC = text.count()
    df = text.flatMap(IDFmap).reduceByKey(add).collect()
    #print "df", df
    dic = process_dic(df, 0, 1)
    cPickle.dump(dic, open("dictionary.dic", 'w'), 2)
    #print CorpusMap("我爱北京天安门", dic, NUM_DOC)
    corpus = text.map(lambda line : CorpusMap(line, dic, NUM_DOC))
    #print "corpus", corpus.collect()
    corpus.saveAsTextFile(sys.argv[2])
    

    
