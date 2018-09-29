import os
import numpy as np
from  collections import Counter

def get_word_list(stop_words):
    class_num = 30
    word_list = [[] for i in range(class_num)]
    all_words = []
    for i in range(class_num):
        with open('word_vector/train_data/'+str(i)+'.txt','r',encoding='utf-8') as f:
            for line in f:
                word = line.split()[0]
                if word not in  stop_words:
                    word_list[i].append(word)
                    if word not in all_words:
                        all_words.append(word)
                       
    return word_list,all_words

def com_tf_part(word_list):
    class_num = len(word_list)
    word_tf ={}
    for i in range(class_num):
        word_tf[i] ={}
        word_nums = len(word_list)
        word_fre = Counter(word_list[i]).most_common()
        for item in word_fre:
            word_tf[i][item[0]] = item[1] / word_nums
    return word_tf
    
    
def com_idf(word_list,all_words):
    word_idf={}
    for word in all_words:
        word_idf[word] = 0
    class_num = 30
    for word in word_idf:
        for item in word_list:
            if word in item:
                word_idf[word] +=1
    for word in word_idf:
        word_idf[word] = np.log10(class_num/(word_idf[word]+1))
    return word_idf

def com_tfidf(word_tf,word_idf):
    word_tfidf={}
    class_num = len(word_tf)
    for i in range(class_num):
        word_tfidf[i] ={}
        for word in word_tf[i]:
            word_tfidf[i][word] = word_tf[i][word] * word_idf[word] 
    return word_tfidf
      
def word_in_class(word,word_list):
    class_num = len(word_list)
    class_table = []
    for i in range(class_num):
        if word in word_list[i]:
            class_table.append(i)
    return class_table


                    
if __name__ == '__main__':
    stop_words =['我','的','了','你','啊','呀']
    word_list,all_words = get_word_list(stop_words)
    word_tf = com_tf_part(word_list)
    word_idf = com_idf(word_list,all_words)
    word_tfidf = com_tfidf(word_tf,word_idf)
    creat_train_data(word_list,word_idf,'word_vector/train_data/train_idf.txt')
    
   #word_idf = con_idf(word_list,word_tf)
   #word_tfidf = con_tfidf(word_tf,word_idf)
   #print(word_list)
   #print(word_tfidf)
   #print(word_idf)