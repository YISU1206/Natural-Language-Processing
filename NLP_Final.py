# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 11:38:49 2018

@author: Helen
"""


import re
import networkx as nx
from itertools import combinations
import sys
import nltk
from nltk import tokenize, pos_tag
from nltk.corpus import stopwords 
from math import log
# read file
def readFile(path):
    with open(path,encoding='ISO-8859-1') as myfile:
        text = myfile.read()
    return text


###################################Part 1: Rank Words 

# split text into sentences
def splitSentence(text):
    sentences = tokenize.sent_tokenize(text)
    return sentences

# get all words in text
def words(text):
    ws = nltk.word_tokenize(text)
    word=[]
    for w in ws:
        if re.match("[a-zA-Z0-9]+", w):
            word.append(w)
    return word


# filter sentence with stop words
def filterStopwords(sentences):
    stopword = stopwords.words('english')
    
    new_sentence = []
    
    for sentence in sentences:
        s=[]
        for wd in words(sentence):
            if wd.lower() not in stopword:
                s.append(wd)
        ss=' '.join(s)
        new_sentence.append(ss)
    return  new_sentence
    
# assign POS tags to the filtered words
def posTag(filtersentence):
    tagged=[]
    for sentence in filtersentence:
        ss=words(sentence.lower())
        tag=pos_tag(ss)
        tagged.append(tag)
    return tagged
    
# only contain nouns and adjectives
def filterTag(tagged):
    ft=[]
    for se in tagged:
        ww=[]
        for w in se:
            if w[1] in ['NN', 'NNP','JJ']:
                ww.append(w[0])
        ft.append(ww)
    return ft

# get all different words to create the vertex in the graph
def difWord(filtered):
    vert=[]
    for sen in filtered:
        for word in sen:
            if word!='':
                vert.append(word.lower())
    return set(vert)

# co-occurrence relation between words, set N=2
def getEdge(filtered):
    edge=[]
    for sen in filtered:
        le=len(sen)
        if le>1:
            for i in range(le)[:-1]:
                edge.append((sen[i].lower(),sen[i+1].lower()))
    return edge

# build graph
def buildGraph(vert, edge):
    # undirected graph
    graph = nx.Graph()
    for vertice in vert:
        graph.add_node(vertice)
    for edge in edge:
        # can change weight (Levenshtein distance)
        graph.add_edge(edge[0], edge[1], weight = 1)
    return graph
    

# return the sorted (decreasing) rank of each word
def Rank_sort(graph):
    # pageRank
    pageRanked = nx.pagerank(graph, alpha=0.85, tol=0.0001, weight='weight')
    # sort key words by decreasing order 
    rank_sort= [(w,pageRanked[w]) for w in sorted (pageRanked,key=pageRanked.get,reverse=True)]

    return rank_sort


# get top N words from dictionary
def getTop(rank_sort, n):
    Key_words= [ pair[0] for pair in rank_sort[:n]]
    
    return Key_words

# check if results are adjacent words in filtered sentences list
# if there are some keywords are adjacent; store them (Phrase)
def joinWords(topWords, ft):
    words = []
    phrase = []
    for se in ft:
        if len(se)>1:
            if se[0] in topWords and se[1] in topWords:
                phrase.append(se[0]+' '+se[1])
                words.extend([se[0],se[1]])
            if len(se)>2:    
                for i in range(len(se))[1:-1]:
                    if se[i-1] not in topWords and se[i] in topWords and se[i+1] in topWords:
                        phrase.append(se[i]+' '+se[i+1])
                        words.extend([se[i],se[i+1]])
    remain = list(set(topWords)-set(words))
    return list(set(phrase))+remain

# change N as an input
def getTopWord(N, rank_sort, ft):

    # check if N is available
    if(N < len(rank_sort)):
        topWords = getTop(rank_sort, N)
      #  signal.alarm(5)

        try:
            modifiedKeyWords = joinWords(topWords, ft)

            return [topWords, modifiedKeyWords]

        except TimeoutException:
            print("Can not get modified key words!")
            return [topWords, None]

    else:
        print("Please try smaller N value.")


###################################Part 2: Rank Sentense 
        
        
# summarization of text  --> rank sentense according to the relationship of sentences
def Similarity(s1,s2):  # s1: adj and n in one sentence (without stopwords)
    if len(s1)>1 and len(s2)>1:
        inter_score = len(set(s1).intersection(set(s2)))/(log(len(s1))+log(len(s2)))
    else:
        inter_score = len(set(s1).intersection(set(s2)))/(log(len(s1)+0.01)+log(len(s2)+0.01))
    return inter_score


def SentenceRank(ft):
    length = len(ft)
    pairs = combinations(range(length), 2)
    scores = [(i, j, Similarity(ft[i], ft[j])) for i, j in pairs]
    scores = filter(lambda x: x[2], scores)
	
    g = nx.Graph()
    g.add_weighted_edges_from(scores)
    pr = nx.pagerank(g)
    return [w for w in sorted (pr,key=pr.get,reverse=True)]
    
def Summary(M, sentences, Senrank):
    Summary=[]
    if len(Senrank)<M:
        for i in Senrank:
            Summary.append(sentences[i])
    else:
        for i in Senrank[:M]:
            Summary.append(sentences[i])
    return ' '.join(Summary)

###################################Part 3: Run functions and output solutions
    
def main(fileName, N, M):

    # read file
    text = readFile(fileName)
    # split text into sentences
    sentences = splitSentence(text)
    # filter sentence with stop words
    new_sentence = filterStopwords(sentences)
    # assign POS tags to the filtered words
    tagged = posTag(new_sentence)
    # only contain nouns and adjectives
    ft = filterTag(tagged)
    # get all different words, get vertices for keyword extraction
    vert = difWord(ft)
    # co-occurrence relation with a window of N words, set N as 2
    edge = getEdge(ft)
    
    graph = buildGraph(vert, edge)

    rank_sort = Rank_sort(graph)

    topWord = getTopWord(N, rank_sort, ft)
        
    S_topWord = topWord[0]
        
    M_topWord = topWord[1]
        
# sentences rank:
        
    Senrank=SentenceRank(ft)
        
       
    summary = Summary(M, sentences, Senrank)
        
#######################
        
# output key words and sentences from text
        
        
    print ("These are top %s key words:" % N)
        
    print ('; '.join(S_topWord))
    print ()
        
    if sorted(M_topWord)!=sorted(S_topWord):
            
        print ("These are modified key words:")
              
        print ('; '.join(M_topWord))
            
        print ()
    if len(summary)!=0:
            
        print ("These are document summary:")
        
        print (summary)
        
        
        
        
        
        
if __name__=="__main__":

    fileName = str(sys.argv[1])
    N = int(sys.argv[2])
    M = int(sys.argv[3])
    main(fileName, N, M)  
        
        
        
       
#################################
#  control run time

# Custom exception class
class TimeoutException(Exception):
    pass

