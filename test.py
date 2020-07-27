import tensorflow as tf
import transformers
#from tokenizers import BertWordPieceTokenizer
#from keras_bert import load_trained_model_from_checkpoint
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Lambda, Bidirectional, LSTM, TimeDistributed, Dense
# from tensorflow.keras.optimizers import Adam
from threading import Thread
from time import time
import multiprocessing
# Detect hardware, return appropriate distribution strategy
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
MAX_LEN = 64
print(tf.test.is_gpu_available())
#with strategy.scope():
#    transformer_layer = (
#        transformers.TFBertModel
#        .from_pretrained('bert-base-chinese')
#    )
#    cls = build_model(transformer_layer, max_len=MAX_LEN)

#print('Load trained Weight')
#cls.load_weights('Classification-model-checkpoint')
#print(cls.summary())


#fast_tokenizer = BertWordPieceTokenizer('bert-zh/vocab.txt', lowercase=False) #"BERT TOKENIZER"

article = '前新北市副市長許志堅涉賄的2件都更案將起死回生。圖為板橋介壽段都更案基地。（記者賴筱桐攝）〔記者賴筱桐／新北報導〕前新北市副市長許志堅2015年因收受建商賄賂，幫助建商快速通過都更案，今年3月判刑10年定讞並入監執行。板橋區介壽段與新店區廣明段2筆都更案在他涉賄案發後，市府皆暫停審議程序，至今停擺3年餘。依法院判決結果，許志堅違反貪污治罪條例不違背職務收受賄賂罪，今年7月初市府都市更新處發文給申請人，通知案件可繼續審議。板橋區介壽段都更案位於中山路1段、中山路1段50巷與民族路街廓，距捷運府中站300公尺，鄰近熱鬧的府中商圈，基地面積約1275平方公尺，其中市有地面積1266平方公尺，公有地佔整體比率99％，土地使用分區為商業區，預計興建地上19層大樓。2013年財政局辦理公開甄選作業，由樂揚建設取得最優申請人資格。請繼續往下閱讀...新店廣明段都更案位於新店區北宜路1段、能仁路口，鄰近新店國小，基地面積5951平方公尺，距捷運新店站600公尺，土地使用分區為住宅區，範圍內多為老舊磚造或鐵皮屋等建物，巷道狹小，寶興建設2008年起申請都更，預計興建地上29層大樓。都市更新處主任秘書李擇仁表示，當初2案進入司法調查階段，因此暫停審議程序，依據高等法院判決結果，許志堅違反貪污治罪條例不違背職務收賄罪，意為行賄事實與案件本身無直接關聯，都更處7月初發函給申請人，告知案件可繼續審議，不須撤案。李擇仁說明，新店廣明段將請申請人再次提送新北市都市更新委員會大會審議，板橋介壽段將請申請人提送小組審查。    不用抽 不用搶 現在用APP看新聞 保證天天中獎　    點我下載APP　    按我看活動辦法'

from src import ClassificationModel
#strategy = tf.distribute.get_strategy()
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

cls = ClassificationModel.load_model()
print('Load trained Weight')
cls.load_weights('bert-Classification-model-64')
print(cls.summary())


print('Load tokenizers')
from tokenizers import BertWordPieceTokenizer
fast_tokenizer = BertWordPieceTokenizer('bert-zh/vocab.txt', lowercase=False)

#import kashgari
from kashgari.tasks.labeling import BiGRU_Model
from kashgari.embeddings import BertEmbedding
from src import NER

print('Load NER Model')
NER_MODLE = BiGRU_Model.load_model('Bert-Chinese_BiGRU_Model')
#NER_MODLE(sequence_length=30)
def op(x):
	y = NER_MODLE.predict_entities(x)
	return y

with tf.device('/cpu:0'):
	cpu = op

with tf.device('/gpu:0'):
	gpu = op

class ThreadWithReturnValue(Thread):
	def __init__(self, group=None, target=None, name=None,
				args=(), kwargs={}, Verbose=None):
		Thread.__init__(self, group, target, name, args, kwargs, Verbose)
		self._return = None
	def run(self):
		if self._Thread__target is not None:
			self._return = self._Thread__target(*self._Thread__args,
												**self._Thread__kwargs)
	def join(self):
		Thread.join(self)
		return self._return
#%%
def predict(article):
	art_enc = ClassificationModel.fast_encode(article, fast_tokenizer, maxlen=MAX_LEN)
	art_dataset = (
   		tf.data.Dataset
   		.from_tensor_slices(art_enc)
   		.batch(1)
	)
	pred = cls.predict(art_dataset) ## predict if the news is about AML
	IND = (pred>0.5).astype('int')
	print(IND)
	if type(IND) is not float and IND == 1:
	## get summarize text
		news = NER.get_summarize(article)

	## NER model
		if len(news) > 0:
			h = round(len(news)/4)

			tf.compat.v1.disable_eager_execution()

			with tf.device('/cpu:0'):
				x = tf.compat.v1.placeholder(name='x', dtype=tf.string)


			def f(session, y, data):
				return session.run(y, feed_dict={x : data})


			with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True, intra_op_parallelism_threads=8)) as sess:
				sess.run(tf.compat.v1.global_variables_initializer())

				coord = tf.train.Coordinator()

				threads = []

				# comment out 0 or 1 of the following 2 lines:
				threads += [ThreadWithReturnValue(target=f, args=(sess, cpu, news[:h]))]
				threads += [ThreadWithReturnValue(target=f, args=(sess, gpu, news[h:]))]

				t0 = time()

				for t in threads:
					t.start()

				coord.join(threads)

				t1 = time()
			print(coord)
			print(t1 - t0)
			prediction = NER.get_names(coord)
		
		else:
		  	prediction = []

	else:
		prediction = []
	#if IND==1:
		
	#prediction = ['aha','danny','jack']


	####################################################
	#prediction = _check_datatype_to_list(prediction)
	return prediction

#%%
if __name__ == "__main__":
	print(predict(article))
  

#%%
# import pandas as pd

# dt = pd.read_csv('/home/wanchu/ESUM/data/all.csv')

# #%%
# from src import NER

# NER.sent_tokenize(dt['news'][1])

# # %%
# from snownlp import SnowNLP
# summm = SnowNLP(dt['news'][192])

# #%%
# import re
# #print("[words]",s.words)
# t_keysen = summm.summary(20)

# def sent_tokenize(x):
#     sents_temp = re.split('(：|:|,|，|。|！|\!|\.|？|\?|、)', x)
#     sents = []
#     for i in range(len(sents_temp)//2):
#         sent = sents_temp[2*i] + sents_temp[2*i+1]
#         sents.append(sent)
#     sents.append(sents_temp[-1])
#     return sents


# def text2array(text, sequence_length):
#     textArr = re.findall('.{' + str(sequence_length) + '}', text)
#     textArr.append(text[(len(textArr) * sequence_length):])
#     return [[c for c in text] for text in textArr]


# news = []
# for t in t_keysen:
# 	sents = sent_tokenize(t)
# 	for s in sents:
# 		ws = SnowNLP(s).words
# 		if any("記者" in s for s in ws):
# 			print('ignore')
# 		else:
# 			print(ws)
# 		#out = text2array(s, sequence_length=len(t))
# 		#news.extend(out)

# # %%
# news

# %%
