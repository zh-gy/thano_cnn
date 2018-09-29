from flask import Flask,request
from Attention_classfier import sentence_predict
import json
import pandas as pd
from fun_table import *

app = Flask(__name__)
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
        result.append({"sentence":i,"label": table_name[int(label)],"prob": str(prob)})
    return pd.DataFrame(result).to_html()
    
if __name__ =='__main__':
    table_name = bank_insertion_dispatcher_table
    #self_function_table = ['id_yes','id_no','busy','dead','missing','complain','contacts_dont_know']
    max_len=10
    with open('word_dict','r',encoding='utf-8') as f:
        word_dict = json.load(f)
    fun_table_name = 'bank_insertion_dispatcher'
    app.run(debug=True)
    