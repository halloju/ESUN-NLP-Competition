import tensorflow as tf
import transformers
#from tokenizers import BertWordPieceTokenizer
#from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

strategy = tf.distribute.get_strategy()

MAX_LEN = 192
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

article = '這是測試文章'

from src import ClassificationModel
strategy = tf.distribute.get_strategy()


cls = ClassificationModel.load_model()
print('Load trained Weight')
cls.load_weights('Classification-model/Classification-model-checkpoint')
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



def predict(article):
	art_enc = ClassificationModel.fast_encode(article, fast_tokenizer, maxlen=MAX_LEN)
	art_dataset = (
   		tf.data.Dataset
   		.from_tensor_slices(art_enc)
   		.batch(1)
	)
	pred = cls.predict(art_dataset) ## predict if the news is about AML
	IND = (pred>0.5).astype('int')

	if type(IND) is not float and IND == 1:
	## get summarize text
		news = NER.get_summarize(article)

	## NER model
		if len(news) > 0:
		  	ners = NER_MODLE.predict_entities(news)

		  	prediction = NER.get_names(ners)
		
		else:
		  	prediction = []

	else:
		prediction = []
	#if IND==1:
		
	#prediction = ['aha','danny','jack']


	####################################################
	#prediction = _check_datatype_to_list(prediction)
	return prediction


if __name__ == "__main__":
	names = predict(article)
	print(names)
  

