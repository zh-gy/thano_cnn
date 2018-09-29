import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pandas import DataFrame
import os



def scale(word_vectors):
    mean = np.mean(word_vectors,axis=0,keepdims=True)
    std= np.std(word_vectors,axis=0,keepdims=True)
    word_vectors_norm= np.divide((word_vectors-mean),std)
    return word_vectors_norm

def get_top_frek_vector(idf_file,word_vectors,word_dict,stop_words,k):
    label_dict ={}
    vocab_fre_list =[]
    label_list = []
    with open(idf_file,'r',encoding='utf-8') as f:
        for line in f:
            item = line.split()
            if float(item[1]) > 0.8:
                label_dict[item[0]] = item[2]  
    for word in label_dict:
        vocab_fre_list.append(word_dict[word])
        label_list.append(label_dict[word])
    
    vocab_fre_list = vocab_fre_list[:k]
    label_list = label_list[:k]
    vector_list =word_vectors[vocab_fre_list]
    
    return vector_list,label_list
  
def dim_reduce(idf_file,word_vectors,word_dict,stop_words,k=500):
    vector_list,label_list = get_top_frek_vector(idf_file,word_vectors,word_dict,stop_words,k)
    vector=scale(vector_list)
    tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    low_dim_embs = tsne.fit_transform(vector)
    return low_dim_embs,label_list
    
def plot_data(low_dim_embs,label_list,batch_size,epoches,embedding_size):
    data ={'x': low_dim_embs[:,0],'y': low_dim_embs[:,1],'label':label_list }
    frame = DataFrame(data)
    color=['black','gray','rosybrown','indianred','firebrick','red','salmon','darksalmon','lightsalmon','sienna','sandybrown','peru','bisque','pink','royalblue','darkorange','tan','darkgoldenrod','wheat','gold','darkkhaki','ivory','olivedrab','yellow','palegreen','darkgreen','darkcyan','deepskyblue','blue','m']
    i = 0
    for name,group in frame.groupby('label'):
        plt.scatter(group['x'],group['y'],c=color[i],marker='o')
        i+=1
    plt.savefig('word_vector/figure/'+str(batch_size)+'_'+str(epoches)+'_'+str(embedding_size)+'_weight.png')
    
def reduce_plot(idf_file,word_vectors,word_dict,stop_words,batch_size,epoches,embedding_size,k=500):
    low_dim_embs,label_list = dim_reduce(idf_file,word_vectors,word_dict,stop_words)
    plot_data(low_dim_embs,label_list,batch_size,epoches,embedding_size)
    
    
    
    
    
