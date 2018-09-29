import numpy as np


def max_heapify(A,i,heap_size):
    left =  (i + 1) * 2 -1
    right = (i + 1) * 2 
    largest = i
    if left < heap_size and A[left][1] > A[largest][1]:
        largest = left
    if right < heap_size and  A[right][1] > A[largest][1]:
        largest = right
    if largest != i:
        A[i],A[largest] = A[largest],A[i]
        max_heapify(A,largest,heap_size)


def build_max_heap(A):
    head_size = len(A)
    for i in range(int(head_size/2),-1,-1):
        max_heapify(A,i,head_size)


def get_top_k(A,reverse_word_dict,k):
    length = len(A)
    heap_size = length
    for i in range(length-1,length-1-k-1,-1):
        A[0],A[i] = A[i],A[0]
        heap_size -= 1
        max_heapify(A,0,heap_size)
    for item in reversed(A[length-k-1:length-1]):
        print(reverse_word_dict[item[0]],item[1])

        
def similarity(word_vector,word_dict,word1,word2):
    #计算两个词之间的相似度
    a = word_vector[word_dict[word1]]
    b = word_vector[word_dict[word2]]
    s1 = np.sqrt(np.sum(np.multiply(a,a)))
    s2 = np.sqrt(np.sum(np.multiply(b,b)))
    s = np.sum(np.multiply(a,b))
    return s/(s1*s2)
    
def most_similarity(word_vector,word,word_dict,reverse_word_dict,k=4):
    #计算top-k最相似次
    vector = word_vector[word_dict[word]][np.newaxis,:]
    inner = np.dot(vector,word_vector.T)
    abs_words = np.sqrt(np.sum(np.power(word_vector.T,2),axis=0,keepdims=True))
    abs_vector = np.sqrt(np.sum(np.power(vector,2)))
    sims =  np.divide(inner,abs_words)
    sims =  sims/abs_vector
    sims_list=[]
    length = sims.shape[1]
    for index in range(length):
        sims_list.append([index,sims[0][index]])
    build_max_heap(sims_list)
    get_top_k(sims_list,reverse_word_dict,k)
    
    