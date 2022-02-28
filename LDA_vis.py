# -*- coding: utf-8 -*-
import pandas as pd
from gensim import corpora, models
import jieba.posseg as stutter
import os.path
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pyLDAvis
import pyLDAvis.gensim_models

# before use this snippet of the code, please change the format to UTF-8
# as in the VSCode editor, change the format in the right-bottom section

def openXLS(iFilename):
    sheet = pd.read_excel(open(iFilename, 'rb'), usecols="A,C,F")
    return sheet

def main():
    # csv filename (complete along with directory)
    csvFilename="csv/weibo_data.xlsx"
    stopwordsFilename="stopwords/cn_stopwords.txt"
    # # output filename
    # txtFilename=f'output/out_weibo_full.txt'

    # set the flag
    flags = ('n', 'nr', 'ns', 'nt', 'v', 'd')
    
    # load the sheet
    sht = openXLS(csvFilename)
    
    # load the stopwords
    stopwords = [line.strip() for line in open(stopwordsFilename, 'r', encoding='utf-8').readlines()]
    stopwords.append("诺贝尔文学奖")
    stopwords.append("网站")
    stopwords.append("网页")
    stopwords.append("发布")
    stopwords.append("先生")
    stopwords.append("微博")
    stopwords.append("链接")  
    
    comments = sht["评论内容"]
    lenComments = len(comments)
    wordList = []

    for i in range(lenComments):
        if 0 == i%1000:
            print(f"task finished at {i}")
        wdsTemptation = []
        for w in stutter.cut(comments[i]):
            if w.flag in flags and w.word not in stopwords:
                wdsTemptation.append(w.word)  
        wordList.append(wdsTemptation)

    dictionary = corpora.Dictionary(wordList)

    ## dictionary post process
    # dictionary.filter_extremes(no_below=20, no_above=0.1)
    # dictionary.compactify()

    # save the dictionary if it not exists
    if not os.path.isfile('dict/weiboNodelLiterature.dict'):
        dictionary.save('dict/weiboNodelLiterature.dict')

    corpus = [dictionary.doc2bow(s) for s in wordList]

    # Set training parameters.
    num_topics = 10

    model = models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics
    )

    model.save('models/v3.model')

    vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'html/LDA_vis_full.html')

if __name__ == '__main__':
    main()
    