import os
import numpy as np
import tensorflow as tf
from creat_train_data import genera_batch,creat_train_data_wordvector,creat_word_dict
from heap import similarity,most_similarity
import numpy as np
from reduce_plot import reduce_plot
import json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_top_frek_vector(word_perc_file,word_vectors,word_dict,stop_words,k):
    label_dict ={}
    vocab_fre_list =[]
    label_list = []
    with open(word_perc_file,'r',encoding='utf-8') as f:
        for line in f:
            item = line.split()
            if float(item[1]) > 0.9:
                label_dict[item[0]] = item[2]  
    for word in label_dict:
        vocab_fre_list.append(word_dict[word])
        label_list.append(label_dict[word])
    
    vocab_fre_list = vocab_fre_list[:k]
    label_list = label_list[:k]
    vector_list =word_vectors[vocab_fre_list]
    
    return vector_list,label_list
  

def train_word_vector(word_perc_file,word_dict,reverse_word_dict,batch_size,epoches,embedding_size,stop_words,class_num=30):
 
    train_data,train_weight,train_label = creat_train_data_wordvector(word_perc_file,word_dict)
    
    #word_dict,reverse_word_dict,train_data,train_weight,train_label = creat_word_dict_idf(idf_file,word_dict_file,vocab_size=1000)
    train_wordvector  = genera_batch(train_data,train_weight,train_label,batch_size)
    
    vocabulary_size =len(word_dict)

    train_examples = train_data.shape[0]
    n_batch = train_examples // batch_size
    
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size],name='data')
    labels = tf.placeholder(tf.int32,shape=[batch_size],name='label')
    one_hot_label = tf.one_hot(labels,depth=class_num,on_value=1,off_value=0)
    weight = tf.placeholder(tf.float32,shape=[batch_size],name='weight')
    
    train_one_input = tf.placeholder(tf.int32)
    
    embeddings = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size]),name='embeddings')
    embed = tf.nn.embedding_lookup(embeddings,train_inputs)
    weights = tf.Variable(tf.truncated_normal([embedding_size,class_num]),name='weights')
    bias = tf.Variable(tf.zeros(shape=[1,class_num]),name='bias')
    
    pretiction = tf.add(tf.matmul(embed,weights),bias)
    cross_entropy = tf.reduce_mean(weight * tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label,logits=pretiction))
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(cross_entropy)
    
    embed_one = tf.nn.embedding_lookup(embeddings,train_one_input)
    softmax = tf.nn.softmax(tf.add(tf.matmul(embed_one,weights),bias))
    predict = tf.argmax(softmax,1)
    
    init = tf.global_variables_initializer() 
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoches):
            for i in range(n_batch):
                batch_inputs,batch_weight,batch_label = next(train_wordvector)
                sess.run(train,feed_dict={train_inputs:batch_inputs,labels:batch_label,weight:batch_weight})
            train_wordvector = genera_batch(train_data,train_weight,train_label,batch_size)
            print(epoch,sess.run(cross_entropy,feed_dict={train_inputs:batch_inputs,labels:batch_label,weight:batch_weight}))
        saver.save(sess,'model/'+'/word_vector/word_vectors_'+str(epoches))
        
        mean_0 = tf.reduce_mean(embeddings,axis = 0,keepdims=True)
        norm_0 = tf.sqrt(tf.reduce_sum(tf.square(embeddings-mean_0),0,keepdims=True))
        normlized_embeddings_0 = (embeddings-mean_0) / norm_0
        mean_1 = tf.reduce_mean(normlized_embeddings_0,axis = 1,keepdims=True)
        norm_1 = tf.sqrt(tf.reduce_sum(tf.square(normlized_embeddings_0-mean_1),1,keepdims=True))
        normlized_embeddings_1 = (normlized_embeddings_0 -mean_1) / norm_1
        embedding = np.array(normlized_embeddings_1.eval())
        print('most_similarity with %s'%'信号')
        most_similarity(embedding,'信号',word_dict,reverse_word_dict,k=6)
        #print('most_similarity with %s'%'逾期')
        #most_similarity(embedding,'逾期',word_dict,reverse_word_dict,k=6)
        #print('most_similarity with %s'%'是')
        #most_similarity(embedding,'是',word_dict,reverse_word_dict,k=6)
        reduce_plot(word_perc_file,embedding,word_dict,stop_words,batch_size,epoches,embedding_size)
        np.save('word_vector/vector/'+str(batch_size)+'_'+str(epoches)+'_'+str(embedding_size),embedding)
        
def analysis_word_vector(word,word_dict,reverse_word_dict,epoches):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/'+'/word_vector/word_vectors_'+str(epoches)+'.meta')
        saver.restore(sess,tf.train.latest_checkpoint('model/'+'word_vector'))
        graph = tf.get_default_graph()
        
        embeddings = graph.get_tensor_by_name('embeddings:0')
        weights = graph.get_tensor_by_name('weights:0')
        bias = graph.get_tensor_by_name('bias:0')
        
        train_one_input = tf.placeholder(tf.int32)
        embed_one = tf.nn.embedding_lookup(embeddings,train_one_input)
        softmax = tf.nn.softmax(tf.add(tf.matmul(embed_one,weights),bias))
        print(sess.run(softmax,feed_dict={train_one_input:np.array([word_dict[word]])}))
        mean_0 = tf.reduce_mean(embeddings,axis = 0,keepdims=True)
        norm_0 = tf.sqrt(tf.reduce_sum(tf.square(embeddings-mean_0),0,keepdims=True))
        normlized_embeddings_0 = (embeddings-mean_0) / norm_0
        mean_1 = tf.reduce_mean(normlized_embeddings_0,axis = 1,keepdims=True)
        norm_1 = tf.sqrt(tf.reduce_sum(tf.square(normlized_embeddings_0-mean_1),1,keepdims=True))
        normlized_embeddings_1 = (normlized_embeddings_0 -mean_1) / norm_1
        embedding = np.array(normlized_embeddings_1.eval())
        #print('most_similarity with %s'%word)
        #most_similarity(embedding,word,word_dict,reverse_word_dict,k=6)
            
if __name__ =='__main__':    
    stop_words =['在','的','了','我','多少']
    idf_file = 'word_vector/train_data/train_idf.txt'
    #word_perc_file = 'word_vector/train_data/train_word_perc.txt'
    word_perc_file = 'word_vector/train_data/train_all_seq.txt'
    word_dict_file ='word_dict.txt'
    #with open('word_dict','r',encoding='utf-8') as f:
    #    word_dict = json.load(f)
    #with open('reverse_word_dict','r',encoding='utf-8') as f:
    #    reverse_word_dict = json.load(f))
    word_dict,reverse_word_dict = creat_word_dict(idf_file,word_dict_file,vocab_size=1000)
   
    batch_size=50
    epoches=401
    embedding_size=15
    class_num = 30
    
    #train_word_vector(word_perc_file,word_dict,reverse_word_dict,batch_size,epoches,embedding_size,stop_words)
    analysis_word_vector('逾期',word_dict,reverse_word_dict,epoches) 
    