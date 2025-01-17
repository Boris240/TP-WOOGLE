from itertools import chain
from collections import defaultdict
import numpy
import pickle
import sys
import copy

with open("tfidf.dict",'rb') as f:
	tfidf = pickle.load(f)
	
with open("tokInfo.dict",'rb') as f:
	tokInfo = pickle.load(f)

with open("pageRank.dict",'rb') as f:
	pageRankDict = pickle.load(f)

print("Normalizing tf idf...",end="")
tfidfNorm = copy.deepcopy(tfidf)

for doc_id, doc_tfidf in tfidfNorm.items(): #TO COMPLETE
    norm = numpy.linalg.norm(list(doc_tfidf.values()))  #TO COMPLETE
    for term, score in doc_tfidf.items():  #TO COMPLETE
        tfidfNorm[doc_id][term] = score / norm  #TO COMPLETE
print("done.")


# Returns the topN documents by token relevance (vector model)
def getBestResults(queryStr, topN, tfidfMatrix):
    query = queryStr.split(" ")
    res = defaultdict(float)
    
    for tok in query:  #TO COMPLETE
        if tok in tfidfMatrix:
            for doc_id, tfidf_score in tfidfMatrix[tok].items():
                res[doc_id] += tfidf_score
    
    for doc_id in res: #TO COMPLETE
        if doc_id in pageRankDict:
            res[doc_id] *= pageRankDict[doc_id]  
    
    sorted_results = sorted(res.items(), key=lambda x: x[1], reverse=True) #TO COMPLETE
    top_results = [doc_id for doc_id, _ in sorted_results[:topN]] #TO COMPLETE
    
    return top_results


# Sorts a list of results according to their pageRank
def rankResults(results):
    rankedResults = sorted(results, key=lambda doc_id: pageRankDict.get(doc_id, 0), reverse=True) #TO COMPLETE
    return rankedResults #TO COMPLETE

def printResults(rankedResults):
	for idx,page in enumerate(rankedResults):
		print(str(idx+1) + ". " + page)


query = "darwin" # or sys.argv[1]
top = 15			 # number of results to show

print("Results for ",query,"\n===========")
results = getBestResults(query,top,tfidf)
printResults(results)

print("\n\nResults after normalization for ",query,"\n===========")
results = getBestResults(query,top,tfidfNorm)
printResults(results)


print("\n\nResults after ranking for ",query,"\n===========")
rankedResults = rankResults(results) #TO COMPLETE
printResults(rankedResults)#TO COMPLETE

#bestPageSimilarity = list(reversed([ searchRes[i] for i in numpy.argsort(searchRes)[-10:] ]))
#bestPageSimilarity


