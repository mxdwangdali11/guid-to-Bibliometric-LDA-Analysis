from ast import Try
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from sklearn.utils import shuffle
from gensim import corpora, models
import pandas as pd
import logging
import pickle
import numpy as np
import os,sys
from matplotlib import pyplot as plt
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from gensim.models.ldamulticore import LdaMulticore

def dumppick(filepath,Year):
    corpus = []   
    tokens = []   
    
    df = pd.read_csv(filepath,sep='\t',encoding="utf-8-sig",error_bad_lines=False)
    df = df[df["Abstract"].isna()!=True]
    df.astype({'year': 'int32'})
    df = df[df.year<Year]
    
    for line in df["Abstract"]:
        corpus.append(line.strip())
    del df
   
    en_stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    
    p_stemmer = PorterStemmer()

  
    logging.info("wenbenyuchuli")
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    for i,text in enumerate(corpus):
        if i%1000==0:
            logging.info(f"{i} line done")
        raw = text.lower()
        token = tokenizer.tokenize(raw)
        stop_remove_token = [word for word in token if (word not in en_stop and len(word)>1)]
        stem_token = [p_stemmer.stem(word) for word in stop_remove_token]
        tokens.append(stem_token)
        # tokens.append(token)
    # print tokens

    logging.info("start")


    logging.info("basic")

    dictionary = corpora.Dictionary(tokens)   
    logging.info("word bags")
    texts = [dictionary.doc2bow(text) for text in tokens]    
    logging.info("finished")


    
    logging.info("start tfidf")
    texts_tf_idf = models.TfidfModel(texts)[texts]     
    pickle.dump(texts, open("text_dtm.pickle","wb"))
    pickle.dump(texts_tf_idf, open("texts_tf_idf_dtm.pickle","wb"))
    pickle.dump(dictionary, open("dictionary.pickle","wb"))

def loadpcik():
    texts = pickle.load(open("text_dtm.pickle","rb"))
    texts_tf_idf = pickle.load(open("texts_tf_idf_dtm.pickle","rb"))
    dictionary = pickle.load(open("dictionary.pickle","rb"))
    return texts, texts_tf_idf,dictionary

#dumppick()
def calc_n_of_lda(filename,Year,start,end):
    dumppick(filename,Year)
    texts, texts_tf_idf, dictionary = loadpcik()
    
    """
    print("**************LSI*************")
    lsi = models.lsimodel.LsiModel(corpus=texts, id2word=dictionary, num_topics=20)    
    texts_lsi = lsi[texts_tf_idf]                
    print(lsi.print_topics(num_topics=20, num_words=10))
    """
        logging.info("**************LDA*************")
    ppl = []
    for num_topics in range(start,end,1):
        texts = shuffle(texts)
        lda = LdaMulticore(corpus=texts,iterations=50, id2word=dictionary, num_topics=num_topics,passes=10,per_word_topics=True)
        #texts_lda = lda[texts_tf_idf]
        # print(lda.print_topics(num_topics=num_topics, num_words=10),file =out)

        # ppl.append(np.exp2(-lda.log_perplexity(texts))
        ppl.append(lda.log_perplexity(texts))
    plt.plot( range(10,60,1),ppl)
    plt.title("num_topics(x) - perplexity(y)")
    plt.savefig("prop.png")
    plt.show()
    return lda, texts, texts_tf_idf, dictionary, ppl

def load_lda(filename,num_topics):
    texts, texts_tf_idf, dictionary = loadpcik()
    lda = LdaMulticore(corpus=texts,iterations=100, id2word=dictionary, num_topics=num_topics,passes=20,per_word_topics=True)
    lda.save("./ldamd/{}tpc+{}".format(num_topics,filename[9:18]))
    return lda, texts, texts_tf_idf, dictionary,

def saveldatpcw(lda,num_topics):
    tpcn = num_topics
    tpcw = pd.DataFrame(columns=[i for i in range(1,11)])
    for i in range(tpcn):
        tpcw.loc[i] = [ w for w,p in lda.show_topic(i)]
    tpcw.to_csv("./newdata/tpcw.csv")


def get_cite_n_dmt(dictionary,citenum=0,):
    citenum=0
    corpus = []   
    tokens = []   
    df = pd.read_csv("pubmed_result_parsed.csv",sep=',',encoding="utf-8-sig")
    df = df[df["cite"]==citenum]
    df = df[df["Abstract"].isna()!=True]
       for line in df["Abstract"]:
        corpus.append(line.strip())
    del df
        en_stop = [ str(i).strip() for i in open("stopwords.txt",encoding="utf-8-sig") ]   # erase stop_words

        p_stemmer = PorterStemmer()

       logging.info("pretreat")
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    for i,text in enumerate(corpus):
        if i%1000==0:
            print(i)
        raw = text.lower()
        token = tokenizer.tokenize(raw)
        stop_remove_token = [word for word in token if (word not in en_stop and len(word)>1)]
        stem_token = [p_stemmer.stem(word) for word in stop_remove_token]
        tokens.append(stop_remove_token)
    texts_cite_n = [dictionary.doc2bow(text) for text in tokens]
    return texts_cite_n

def getallcited2tpc(lda,texts,filename,Year):
    tpc1 = []
    tpc2 = []
    for i in texts:
        tpc = lda.get_document_topics(i)
        tpc = sorted(tpc,key=lambda x:-x[2])
        tpc1.append(tpc[0][0])
        if len(tpc)>1:
            tpc2.append(tpc[2][0])
        else:
            tpc2.append(lda.num_topics+1)
    df = pd.read_csv(filename,sep='\t',encoding="utf-8-sig")
    df = df[df["Abstract"].isna()!=True]
    df = df[df.year<Year]
    df["tpc1"] = tpc1
    df["tpc2"] = tpc2
    df.to_csv(filename.replace(".csv","_with_topic.csv"),sep='\t',encoding="utf-8-sig")
    return tpc1,tpc2

def grap(tpc1,tpc2,tpcn,filename):
    from collections import  Counter
    CC = Counter(tpc1)
    import networkx as nx
    G = nx.Graph()
    for i in range(tpcn):
        G.add_node(i,num=CC[i])
    #edgscount = Counter([(i,j) for i,j in zip(tpc1,tpc2)])
    #for edgs,count in edgscount.items()
    edgeslist = [(i,j)   for i,j in zip(tpc1,tpc2) if j<tpcn]
    G.add_edges_from(edgeslist,Weight=0)
    for i,j in zip(tpc1,tpc2):
        if j >= tpcn:
            continue
        G.edges[i,j]["Weight"]+=1
    nx.write_graphml(G,filename.replace(".csv",".graphml"),encoding="utf8")
    return G

def Grap_Add_tpcname():
    import pandas as pd
    c = pd.read_excel("glioblastoma\\topics-gbm.xlsx")
    pic = {tpcid:name for tpcid,name in  zip( range(50),c["topics"])}
    import networkx as nx
    g = nx.read_graphml("glioblastoma\\Glioblastoma_50.graphml")
    for i in range(50):
        g.node[str(i)]["name"] = pic[i]
    nx.write_graphml(g,"glioblastoma\\Glioblastoma_50.graphml_addname.graphml",encoding="utf8")

def main(Year):
    filename = "total.csv"

    if not os.path.exists("./newdata"):
        os.mkdir("./newdata")
    if not os.path.exists("./ldamd"):
        os.mkdir("./ldamd")
    start, end = map(int, str(input("Enter calculation subject range comma-separated (example 10,60): ")).split(","))
    calc_n_of_lda(filename,Year, start, end)
    num_topics = int(input("Enter the number of topics："))
    lda, texts, texts_tf_idf, dictionary = load_lda(filename,num_topics)

    saveldatpcw(lda,num_topics)
    logging.info("LDA finish")
    logging.info("in each publication")
    tpc1,tpc2 = getallcited2tpc(lda,texts,filename,Year)
    
    logging.info("fig formation")
    grap(tpc1,tpc2,num_topics,filename)
    
if __name__ == "__main__":
    Year = 2022
main(Year)
