
from flask import Flask
from flask import request
from flask import jsonify
import datetime
import hashlib
import numpy as np
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

app = Flask(__name__)
####### PUT YOUR INFORMATION HERE ########
CAPTAIN_EMAIL = 'wanchu.lin133@gmail.com'#
SALT = 'Taiwan-No.1'                     #
##########################################


###########################    MODEL     ################################
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer
from src import ClassificationModel
import kashgari
from kashgari.tasks.labeling import BiGRU_Model
from kashgari.embeddings import BertEmbedding
from src import NER

from kashgari.logger import logger
logger.disabled = True
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

## AML Classification Model
MAX_LEN = 256
cls = ClassificationModel.load_model(MAX_LEN)
print('Load trained Weight')
cls.load_weights('bert-Classification-Model-256')

## Sents chosen
m_len = 64
cents = ClassificationModel.load_model(m_len)
print('Load trained Weight')
cents.load_weights('Sents-Classification-model-64')

## NER Model
fast_tokenizer = BertWordPieceTokenizer('bert-zh/vocab.txt', lowercase=False) #"BERT TOKENIZER"
NER_MODLE = BiGRU_Model.load_model('Bert-Chinese_BiGRU_Model-100')
print('Load NER Model')
kashgari.config.use_cudnn_cell = True


##########################################################################


def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def predict(article):
    art_enc = ClassificationModel.fast_encode(article, fast_tokenizer, maxlen=MAX_LEN)
    art_dataset = (
   		tf.data.Dataset
   		.from_tensor_slices(art_enc)
   		.batch(1)
	)
    pred = cls.predict(art_dataset) ## predict if the news is about AML
    IND = (pred>0.5).astype('int')
	# print(IND)
    if type(IND) is not float and IND == 1:
        sents = NER.get_summarize(article)
		# print(sents)
        sents_enc = ClassificationModel.sents_encode(sents, fast_tokenizer, maxlen=m_len) 
        sents_rel = cents.predict(sents_enc) ## select relevant sentances
        IND_sents = (sents_rel>0.5).astype('int')
	## NER model
        news=[]
        for i, ind in enumerate(IND_sents):
            if ind ==1:
                sent_split = NER.sent_tokenize(sents[i])
                for o in sent_split:
                    out = NER.text2array(o, sequence_length=len(o))
                    news.extend(out)
        
        if len(news)>0:
            ners = NER_MODLE.predict_entities(news)
            prediction = NER.get_names(ners)
        else:
            prediction = []
    else:
        prediction = []
    
    prediction = _check_datatype_to_list(prediction)
    
    return prediction

def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.
        
    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)  
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)  
    #esun_timestamp = data['esun_timestamp'] #自行取用
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    
    try:
        answer = predict(data['news'])
    except:
        raise ValueError('Model error.')        
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})

@app.after_request
def after_request(response):
    timestamp = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info('%s %s %s', timestamp, request.get_json(),response.get_json())
    return response
# def log_request_info():
#     # app.logger.info('Headers: %s', request.headers)
#     app.logger.info('Body: %s', data)
#     app.logger.info('Answer: %s', answer)

if __name__ == "__main__":
    handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=10)
    logger = logging.getLogger('tdm')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)   
    app.run(host='0.0.0.0', port=8080, debug=False)
