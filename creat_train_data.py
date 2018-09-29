import os
import pandas as pd
import re
import jieba
import shutil
import collections
import numpy as np
from numpy import random
import json


from tf_idf import get_word_list ,com_idf

def creat_label_sentence(file_name,sheet_list):
    label_sentence ={}
    for sheet in sheet_list:
        data_frame = pd.read_excel(file_name,sheet)
        data_frame = data_frame.fillna('')
        for key in data_frame.keys():
            if key not in label_sentence:
                label_sentence[key] = [ re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）\n]+", "",s.strip()) for s in data_frame[key] if len(s) > 0]
    return label_sentence
    
def creat_map():
    map_dict ={}
    map_dict['id_yes'] = 0
    map_dict['id_no'] = 1
    map_dict['busy']= 2
    map_dict['dead']= 3
    map_dict['missing']= 4
    map_dict['complain']= 5
    map_dict['contacts_dont_know']= 6
    map_dict['contacts_know']= 7
    map_dict['will_repay']=8
    map_dict['already_repay']=9
    map_dict['cant_repay']=10
    map_dict['bill_dispute']=11
    map_dict['negotiation']=12
    map_dict['will_repay_date']=13
    map_dict['do_check_bill']=14
    map_dict['dont_check_bill']=15
    map_dict['which_card']=16
    map_dict['amount']=17
    map_dict['how_to_repay']=18
    map_dict['phone_notification']=19
    map_dict['installment']=20
    map_dict['cant_use']=21
    map_dict['robot']=22
    map_dict['overdue_confirm']=23
    map_dict['interest']=24
    map_dict['detail']=25
    map_dict['minimum_payment']=26
    map_dict['repayment_date']=27
    map_dict['customer_service']=28
    map_dict['already_contact_service']=29
    map_dict['id_yes_again'] = map_dict['id_yes']
    map_dict['id_no_again'] = map_dict['id_no']
    map_dict['cant_repay_again'] = map_dict['cant_repay']
    map_dict['negotiation_success']= map_dict['will_repay']
    map_dict['negotiation_failure']= map_dict['cant_repay']
    map_dict['cant_repay_date']=map_dict['cant_repay']
    map_dict['check_repay_date'] = map_dict['will_repay']
    map_dict['complain_break'] = map_dict['complain']
    map_dict['busy_break'] = map_dict['busy']
    return map_dict
    
    
def creat_train_class(label_sentence,map_dict,word_dict,path_dir,file_name,rm=True):
    train_dict ={}
    jieba.load_userdict('user_dict.txt')
    for key in map_dict:
        train_dict[map_dict[key]]=[]
    for key in map_dict:
        for sentence in label_sentence[key]:
            train_dict[map_dict[key]].append(jieba.lcut(sentence))
    if os.path.exists(path_dir) and rm==True:
        shutil.rmtree(path_dir)
    os.makedirs(path_dir)
    for key in train_dict:
        with open(os.path.join(path_dir,file_name),'a',encoding='utf-8') as f:
            for sentence in train_dict[key]:
                s =''
                for word in sentence:    
                    if word in word_dict:
                        s = s +'  ' +word
                if len(s.strip())>0:
                    s =s + '   ' + str(key) + '\n'
                    f.write(s)
    
def creat_map_file(label_sentence,map_dict,path_dir='word_vector/train_data',rm=True):
    train_dict ={}
    jieba.load_userdict('user_dict.txt')
    for key in map_dict:
        train_dict[map_dict[key]]=[]
    for key in map_dict:
        for sentence in label_sentence[key]:
            train_dict[map_dict[key]].extend(jieba.lcut(sentence))
    #if os.path.exists(path_dir) and rm==True:
    #    shutil.rmtree(path_dir)
    #os.makedirs(path_dir)
    for key in train_dict:
        file_name = str(key) + '.txt'
        with open(os.path.join(path_dir,file_name),'w',encoding='utf-8') as f:
            for word in train_dict[key]:
                f.write(word)
                f.write('\n')
        file_name ='train_all.txt'
        with open(os.path.join(path_dir,file_name),'a',encoding='utf-8') as af:
            for word in train_dict[key]:
                af.write(word+ '   '+ str(key))
                af.write('\n')
                
    
   
def creat_train_file(word_list,word_idf,out_file):
    class_num = len(word_list)
    with open(out_file,'w',encoding='utf-8') as f:
        for i in range(class_num):
            for word in set(word_list[i]):
                #if word_idf[word] > idf_thren:
                s = word + '   '+ str(word_idf[word]) + '   ' + str(i) +'  \n'
                f.write(s)


def creat_word_dict(train_file,word_dict_file,vocab_size):
    words =[]
    with open(train_file,'r',encoding='utf-8') as f:
        for line in f:
            words.append(line.split(' ')[0])
    word_fre_table = collections.Counter(words).most_common(vocab_size-1)
    i = 0
    word_dict= {}
    reverse_word_dict ={}
    with open(word_dict_file,'w',encoding='utf-8') as f:
        for item in word_fre_table:
            f.write(item[0])
            f.write('  ')
            f.write(str(i))
            f.write('\n')
            word_dict[item[0]]=i
            reverse_word_dict[i] = item[0]
            i+=1
    with open('word_dict','w',encoding='utf-8') as f:
        json.dump(word_dict,f)
    with open('reverse_word_dict','w',encoding='utf-8') as f:
        json.dump(reverse_word_dict,f)
    return word_dict,reverse_word_dict
    
def creat_train_file_class_perc(word_list,word_perc_file,word_dict,word_idf,perc_thren,idf_thren):
    #统计每个词出现在不同类中词频的百分比
    class_num = len(word_list)
    vocab_size = len(word_dict)
    vocab_class =[[0 for i in range(class_num)] for j in range(vocab_size)]
    for i in range(class_num):
        for word in word_list[i]:
            if word in word_dict:
                vocab_class[word_dict[word]][i] += 1            
    #word_idf_vector =[0 for i in range(vocab_size)]
    #for word in word_dict:
    #    word_idf_vector[word_dict[word]] = word_idf[word]
    #word_idf_vector = np.array(word_idf_vector)[:,np.newaxis]
    vocab_class = np.array(vocab_class)
    vocab_class = vocab_class / np.sum(vocab_class,axis=1,keepdims=True)
    #vocab_class = vocab_class * word_idf_vector
    with open(word_perc_file,'w',encoding='utf-8') as f:
        for i in range(class_num):
            for word in set(word_list[i]):
                if word in word_dict and vocab_class[word_dict[word]][i] >= perc_thren and word_idf[word] > idf_thren:
                    s = word + '   '+ str(vocab_class[word_dict[word]][i]) + '   ' + str(i) +'  \n'
                    f.write(s)

def creat_train_word_vector_file_all_seq(word_dict,word_idf,perc_thren,idf_thren):
    file_name ='word_vector/train_data/train_all.txt'
    out_file_name = 'word_vector/train_data/train_all_seq.txt'
     #统计每个词出现在不同类中词频的百分比
    class_num = len(word_list)
    vocab_size = len(word_dict)
    vocab_class =[[0 for i in range(class_num)] for j in range(vocab_size)]
    for i in range(class_num):
        for word in word_list[i]:
            if word in word_dict:
                vocab_class[word_dict[word]][i] += 1            
    #word_idf_vector =[0 for i in range(vocab_size)]
    #for word in word_dict:
    #    word_idf_vector[word_dict[word]] = word_idf[word]
    #word_idf_vector = np.array(word_idf_vector)[:,np.newaxis]
    vocab_class = np.array(vocab_class)
    vocab_class = vocab_class / np.sum(vocab_class,axis=1,keepdims=True)
    #vocab_class = vocab_class * word_idf_vector
    
    with open(out_file_name,'w',encoding='utf-8') as w:
        with open(file_name,'r',encoding='utf-8') as f:
            for line in f:
                item = line.split()
                if item[0] in word_dict and vocab_class[word_dict[item[0]]][int(item[1])] >= perc_thren and word_idf[item[0]] > idf_thren:
                    s = item[0] + '  '
                    s  = s + str(vocab_class[word_dict[item[0]]][int(item[1])]) + '  '
                    s  = s + item[1]
                    s  = s + '\n'
                    w.write(s)
    #print(vocab_class)
def creat_train_data_wordvector(train_file,word_dict):
    train_data = []
    train_label = []
    train_weight = []
    n = len(word_dict)
    with open(train_file,'r',encoding='utf-8') as f:
        for line in f:
            item = line.split()
            if item[0] in word_dict:
                train_data.append(word_dict[item[0]])
                train_weight.append(item[1])
                train_label.append(item[2])
    train_data = np.array(train_data)
    train_weight = np.array(train_weight)
    train_label = np.array(train_label)
    return train_data,train_weight,train_label

def creat_classfier_train_data(word_dict,path_dir,file_name,max_len = 10):
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
                m = max_len % local_len
                for i in range(n):
                    for j in range(local_len):
                        local_sen.append(word_dict[item[j]])
                rand = np.random.randint(0,local_len,m)
                for i in rand:
                    local_sen.append(word_dict[item[i]])
            train_data.append(local_sen)
            train_label.append(item[-1])
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    return train_data,train_label
    
def genera_batch_classfier(train_data,train_label,batch_size):
    len_example = train_data.shape[0]
    seq = np.arange(len_example)
    random.shuffle(seq)
    train_data = train_data[seq]
    train_label = train_label[seq]
    n_batch = len_example // batch_size
    for i in range(n_batch):
        yield(train_data[i*batch_size:(i+1)*batch_size],train_label[i*batch_size:(i+1)*batch_size])
                  
def genera_batch(train_data,train_weight,train_label,batch_size):
    len_example = train_data.shape[0]
    seq = np.arange(len_example)
    random.shuffle(seq)
    train_data = train_data[seq]
    train_wieght = train_weight[seq]
    train_label = train_label[seq]
    n_batch = len_example // batch_size
    for i in range(n_batch):
        yield(train_data[i*batch_size:(i+1)*batch_size],train_weight[i*batch_size:(i+1)*batch_size],train_label[i*batch_size:(i+1)*batch_size])
if __name__ == '__main__':
    file_name = 'NLU语料与测试.xlsx'
    sheet_list = ['self_function','self_function2','whether_contacts_know1','whether_repay','whether_repay_2','whether_check_bill','whether_check_date',
                 'whether_negotiation','whether_confirm_bill','bank_insertion_dispatcher']
    label_sentence = creat_label_sentence(file_name,sheet_list)
    del label_sentence['Unnamed: 6']
    del label_sentence['Unnamed: 7']
    del label_sentence['Unnamed: 8']
    del label_sentence['Unnamed: 9']
    del label_sentence['Unnamed: 10']
    del label_sentence['bill_dispute（结果和此列不匹配）']
    
    stop_words =['在','的','了','我','多少']
    idf_thren = 0.45
    perc_thren = 0.1
    vocab_size = 1000
    map_dict = creat_map()
    idf_file = 'word_vector/train_data/train_idf.txt'
    word_perc_file = 'word_vector/train_data/train_word_perc.txt'
    #word_perc_file = 'word_vector/train_data/train_all_seq.txt'
    word_dict_file ='word_dict.txt'
    path_dir = 'classfier/train_data'
    file_name ='train_classfier.txt'
    creat_map_file(label_sentence,map_dict,rm=True)
    word_list,all_words = get_word_list(stop_words)
    word_idf = com_idf(word_list,all_words)
    
    creat_train_file(word_list,word_idf,idf_file)
    word_dict,reverse_word_dict = creat_word_dict(idf_file,word_dict_file,vocab_size)
    
    
    
    #creat_train_file_class_perc(word_list,word_perc_file,word_dict,word_idf,perc_thren,idf_thren)
    
    #train_data,train_weight,train_label =creat_train_data_wordvector(word_perc_file,word_dict)
    creat_train_word_vector_file_all_seq(word_dict,word_idf,perc_thren,idf_thren)
    
    
    creat_train_class(label_sentence,map_dict,word_dict,path_dir,file_name)
    #train_data,train_label = creat_classfier_train_data(word_dict,path_dir,file_name)
    #train_classfier = genera_batch_classfier(train_data,train_label,batch_size=10)
    #print(next(train_classfier))
    #creat_train_word_vector(label_sentence,map_dict)
    #train_wordvector  = genera_batch(train_data,train_weight,train_label,batch_size=10)
    #print(next(train_wordvector))
    

   