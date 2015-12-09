#coding:gbk

import sys
import Pyro4
import jieba
import cPickle
import math
import os


class LdaModel:
    def __init__(self, dic_name, file_name):
        self.beta = self.load_beta(file_name)
        self.dic = self.load_dic(dic_name)
 
    def load_dic(self, dic_name):
        return cPickle.load(open(dic_name))
        
    def load_beta(self, file_name):
        res = {}
        i = 1
        for line in open(file_name):
            if i % 10000 == 0:
                print i
            i = i+1
            line = line.strip().split(' ')
            key = int(line[0])
            value = [float(x) for x in line[1:]]
            res[key] = value
        return res

    def get_topic(self, data):
        res = []
        for ele in data:
            value = self.beta.get(ele[0])
            if not value:
                continue
            if not res:
                res = value
                continue
            else:
                res = [res[i] + value[i] for i in range(len(value))]
        return res
    
    def get_corpus(self, line):
        line = jieba.cut(line.strip(), cut_all = True)
        res = {}
        for ele in line:
            if ele in res:
                res[ele] = res[ele] + 1
            else:
                res[ele] = 1
        return [(self.dic[ele][0], self.dic[ele][1] * res[ele]) for ele in res if ele in self.dic]
    
    def predict(self, line):
        corpus = self.get_corpus(line)
        topic = self.get_topic(corpus)
        return list(enumerate(topic))

    def cos_relate(self, r1, r2):
        if not r1 or not r2:
            return 0.0
        num1 = math.sqrt(sum([x * x for x in r1]))
        num2 = math.sqrt(sum([x * x for x in r2]))
        num3 = sum(r1[i] * r2[i] for i in range(len(r1)))
        return num3 / (num1 * num2)
    
if __name__ == "__main__":
    
    PORT = 8996
    HOST = os.popen('hostname -i').read().strip()
    
    model = LdaModel(sys.argv[1], sys.argv[2])
    daemon = Pyro4.Daemon(host = HOST, port = PORT)
    daemon.register(model, "ldamodel")
    print "starting loop"
    daemon.requestLoop()
    
    
    
