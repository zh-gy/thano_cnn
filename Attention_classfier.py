import tensorflow as tf
import numpy as np
import json
from creat_train_data import creat_word_dict,creat_map
import os
import jieba
import pandas as pd

from flask import Flask,request
from fun_table import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

def get_attention(word_perc_file,word_vectors,word_dict,class_num=30,thren=0.5):
    attention_label = [[] for i in range(class_num)]
    with open(word_perc_file,'r',encoding='utf-8') as f:
        for line in f:
            item = line.split()
            if float(item[1]) >= thren:
                attention_label[int(item[2])].append(word_dict[item[0]])
    attention_vector = []
    for i in range(class_num):
        attention_vector.append(np.mean(word_vectors[attention_label[i]],axis=0))
    attention_vector = np.array(attention_vector)
    return attention_vector

def creat_train_file(all_train_file,word_dict,map_dict,fun_table,fun_table_name ,path_dir):
    length = len(fun_table)
    #建立本地意图与整体意图的关系
    intent_map = {} 
    for i in range(length):
        #intent_map[map_dict[fun_table[i]]] = i
        intent_map[i] = map_dict[fun_table[i]]
    with open(os.path.join(path_dir,fun_table_name),'w',encoding='utf-8') as w:
        with open(all_train_file,'r',encoding='utf-8') as r:
            for line in r:
                item = line.split()
                length = len(item)
                for key in intent_map:
                    if int(item[-1]) == intent_map[key]:
                        s = ''
                        for i in range(length - 1):
                            s = s + item[i] + '  '
                        s = s + str(key) +'\n'
                        w.write(s)
    return intent_map

def get_local_attention(all_attention,intent_map):
    #生成本地attention
    local_attention = []
    for key in intent_map:
        local_attention.append(all_attention_vector[intent_map[key]])
    local_attention = np.array(local_attention)
    return local_attention
      
def creat_classfier_train_data(word_dict,path_dir,file_name,max_len=10):
    #补齐不够长的部分，并使用生成器生产batch训练数据
    train_data=[]
    train_label =[]
    vocab_size= len(word_dict)
    with open(os.path.join(path_dir,file_name),'r',encoding='utf-8') as f:
        for line in f:
            item = line.split()
            local_sen =[]
            local_len = len(item) - 1
            if local_len >= max_len:
                for i in range(max_len):
                    local_sen.append(word_dict[item[i]])
            else:
                n = max_len // local_len
                for i in range(n):
                    for j in range(local_len):
                        local_sen.append(word_dict[item[j]])
            train_data.append(local_sen)
            train_label.append(int(item[-1]))  
    return train_data,train_label

def get_train_data(train_data,train_label,max_len=10):
    for local_sen in train_data:
        local_len = len(local_sen)
        local_tab = max_len % local_len
        assert local_tab < local_len
        rand = np.random.randint(0,local_len,local_tab)
        for i in rand:
            local_sen.append(local_sen[i])                                                
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    return train_data,train_label
        

def train_data_generator(train_data,train_label,max_len,batch_size):
    for local_sen in train_data:
        local_len = len(local_sen)
        local_tab = max_len % local_len
        assert local_tab < local_len
        rand = np.random.randint(0,local_len,local_tab)
        for i in rand:
            local_sen.append(local_sen[i])                                                
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    len_example = train_data.shape[0]
    seq = np.arange(len_example)
    np.random.shuffle(seq)
    train_data = train_data[seq]
    train_label = train_label[seq]
    n_batch = len_example // batch_size
    for i in range(n_batch):
        yield(train_data[i*batch_size:(i+1)*batch_size],train_label[i*batch_size:(i+1)*batch_size])
   
def intent_classfier_train(word_vectors,local_attention,train_data,train_label,fun_table_name,max_len,epoches,batch_size):
    class_num = local_attention.shape[0]
    embedding_size = word_vectors.shape[1]
    train_examples = len(train_data)
    n_batch = train_examples // batch_size
    
    #训练数据生成器
    data_generator = train_data_generator(train_data,train_label,max_len,batch_size)
    
    #placeholder
    with tf.name_scope('placeholder'):
        inputs = tf.placeholder(tf.int32,shape=[batch_size,max_len],name='sentence')
        lables = tf.placeholder(tf.int32,shape=[batch_size],name='label')
    
    one_hot_label = tf.one_hot(lables,depth=class_num,on_value=1,off_value=0)
    
    #embedding
    with tf.variable_scope('embedding'):
        word_vector_embeddings = tf.Variable(word_vectors,trainable=False,dtype=tf.float32,name='word_vector')
        attention_embeddings = tf.Variable(local_attention,dtype= tf.float32,name='attention')
    
    #Variable
    with tf.variable_scope('classfier'):
        weights = tf.Variable(tf.truncated_normal([embedding_size,class_num]),name='weights')
        bias = tf.Variable(tf.zeros(shape=[1,class_num]),name='bias')
  
    
    #训练       
    merge_vectors = []
    for i in range(batch_size):
        word_vector_embed = tf.nn.embedding_lookup(word_vector_embeddings,inputs[i])
        attention_embed = tf.nn.embedding_lookup(attention_embeddings,lables[i])
        inner = tf.reduce_sum(word_vector_embed * attention_embed,axis=1)
        attention = tf.nn.softmax(inner)
        attention = tf.expand_dims(attention,1)
        merge_vector = tf.reduce_sum(word_vector_embed* attention,axis=0)
        merge_vectors.append(merge_vector)
    prediction = tf.matmul(merge_vectors,weights) + bias
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label,logits=prediction))
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(cross_entropy)
    
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)
       
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoches):
            for step in range(n_batch):
                batch_data,batch_label = next(data_generator)
                sess.run(train,feed_dict ={inputs:batch_data,lables:batch_label})
            data_generator = train_data_generator(train_data,train_label,max_len,batch_size)
            if epoch % 40 == 0:
                print(epoch,sess.run(cross_entropy,feed_dict ={inputs:batch_data,lables:batch_label}))         
        saver.save(sess,'model/'+fun_table_name+'/intent')    

                
def intent_classfier_test(test_data,test_label,fun_table_name):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/'+fun_table_name+'/intent.meta')
        saver.restore(sess,tf.train.latest_checkpoint('model/'+fun_table_name))
        graph = tf.get_default_graph()
        word_vector_embeddings = graph.get_tensor_by_name("embedding/word_vector:0")
        attention_embeddings = graph.get_tensor_by_name('embedding/attention:0')
        weights = graph.get_tensor_by_name('classfier/weights:0')
        bias = graph.get_tensor_by_name('classfier/bias:0')
        
        all_prob = []
        all_label = []
        test_size = test_data.shape[0]
    
        test_inputs =tf.placeholder(tf.int32,shape=[None,max_len],name='test_inputs')
        test_labels = tf.placeholder(tf.int32,shape=[None],name='test_label')
        
        for i in range(test_size):
            one_word_vector_embed = tf.nn.embedding_lookup(word_vector_embeddings,test_inputs[i])
            inner = tf.matmul(one_word_vector_embed,tf.transpose(attention_embeddings))
            inner = tf.nn.softmax(inner,axis=0)
            merge_vector = tf.matmul(tf.transpose(inner),one_word_vector_embed)
            prob_raw = tf.nn.softmax(tf.add(tf.matmul(merge_vector,weights),bias),axis=1)
            prob = tf.reduce_max(prob_raw,axis=1)
            label = tf.argmax(prob_raw,axis=1)
            all_prob.append(tf.reduce_max(prob)) 
            all_label.append(tf.slice(label,[tf.argmax(prob)],[1])[0])
        
        all_label = tf.cast(tf.stack(all_label),tf.int32)
        correct_prediction = tf.equal(test_labels,all_label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        print('predict_labels',sess.run(all_label,feed_dict ={test_inputs:test_data,test_labels:test_label})) 
        print('test_labels',sess.run(test_labels,feed_dict ={test_inputs:test_data,test_labels:test_label})) 
        print('accuracy',sess.run(accuracy,feed_dict ={test_inputs:test_data,test_labels:test_label})) 

def intent_classfier_predict(pre_sen,fun_table_name):

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/'+fun_table_name+'/intent.meta')
        saver.restore(sess,tf.train.latest_checkpoint('model/'+fun_table_name))
        graph = tf.get_default_graph()
        word_vector_embeddings = graph.get_tensor_by_name("embedding/word_vector:0")
        attention_embeddings = graph.get_tensor_by_name('embedding/attention:0')
        weights = graph.get_tensor_by_name('classfier/weights:0')
        bias = graph.get_tensor_by_name('classfier/bias:0')
        
        one_input = tf.placeholder(tf.int32,shape=[None],name='one_put')
        one_word_vector_embed = tf.nn.embedding_lookup(word_vector_embeddings,one_input)
        
        inner = tf.matmul(one_word_vector_embed,tf.transpose(attention_embeddings))
        inner = tf.nn.softmax(inner,axis=0)
        merge_vector = tf.matmul(tf.transpose(inner),one_word_vector_embed)
        prob_raw = tf.nn.softmax(tf.add(tf.matmul(merge_vector,weights),bias),axis=1)
        
        prob = tf.reduce_max(prob_raw,axis=1)
        
        label = tf.argmax(prob_raw,axis=1)
        prob = sess.run(prob,feed_dict={one_input:pre_sen})
        label = sess.run(label,feed_dict={one_input:pre_sen})
        
        prob = np.max(prob)
        label = label[np.argmax(prob)]
        #print(sess.run(inner,feed_dict={one_input:pre_sen}))
        return prob,label

def sentence_predict(sentence,fun_table_name,word_dict,max_len):
    jieba.load_userdict('user_dict.txt')
    words_list = jieba.lcut(sentence)
    pre_sen = []
    for word in words_list:
        if word in word_dict:
            pre_sen.append(word_dict[word])
    if len(pre_sen) > 0:
        pre_sen = np.array(pre_sen)
        prob,label = intent_classfier_predict(pre_sen,fun_table_name)
        return prob,label
        

    
classifier = lambda x: sentence_predict(x, fun_table_name, word_dict, max_len)
    
@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    sentence = data.pop("sentence", None)
    if not isinstance(sentence, list):
        sentence = [sentence]     
    result = []
    for i in sentence:
        prob, label = classifier(i)
        result.append({"prob": str(prob), "label": self_function_table[int(label)],"sentence":i})
    return pd.DataFrame(result).to_html()
                 
if __name__ =='__main__':

    word_perc_file = 'word_vector/train_data/train_word_perc.txt'
    word_vectors_file ='word_vector/vector/50_601_15.npy'
    all_train_file = 'classfier/train_data/train_classfier.txt'
    path_dir='classfier/train_data/'
    max_len=10
    batch_size=20
    epoches = 1001
    
    with open('word_dict','r',encoding='utf-8') as f:
        word_dict = json.load(f)
    print(len(word_dict))
    word_vectors = np.load(word_vectors_file)
    all_attention_vector = get_attention(word_perc_file,word_vectors,word_dict)

    map_dict = creat_map()
    fun_table = bank_insertion_dispatcher_table
    fun_table_name = 'bank_insertion_dispatcher'
    train_file_name = 'train_' + fun_table_name + '.txt' 
    
    intent_map = creat_train_file(all_train_file,word_dict,map_dict,fun_table,train_file_name,path_dir)
    local_attention = get_local_attention(all_attention_vector,intent_map)
    
    train_data,train_label = creat_classfier_train_data(word_dict,path_dir,train_file_name)
    
    intent_classfier_train(word_vectors,local_attention,train_data,train_label,fun_table_name,max_len,epoches,batch_size)
    
    test_data,test_label = get_train_data(train_data,train_label,max_len=10)
    intent_classfier_test(test_data,test_label,fun_table_name)
    
    #table_list =[self_function_table,whether_contacts_know_table,whether_repay_table,whether_check_date_table,whether_negotiation_table,bank_insertion_dispatcher_table,whether_check_bill_table,all_intent_table]
    #table_name_list=['self_function','whether_contacts_know','whether_repay','whether_check_date','whether_negotiation','bank_insertion_dispatcher','whether_check_bill','all_intent']
   
    #pre_sen = np.array([62])
    #prob,label = intent_classfier_predict(pre_sen,fun_table_name)
    #print(prob,label)
    
    #sentence='我不是他'
    #prob,label = sentence_predict(sentence,fun_table_name,word_dict,max_len)
    # print(prob,label)
    #pre_sen = np.array([62,62,62,62,62,62,62,62,62,62])
    #prob,label = intent_classfier_predict(pre_sen)
    #print(prob,label)
   
        
    #app.run(debug=True)

    
    