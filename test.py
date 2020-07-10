import tensorflow as tf
import transformers
#from tokenizers import BertWordPieceTokenizer
#from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

strategy = tf.distribute.get_strategy()


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

article = '發生在民國89年的泰國海洛因磚走私案，檢調在高雄縣大寮區查獲海洛因磚60塊、重達21公斤。主嫌鍾增林逃亡17年後返台遭逮，二審遭判無期徒刑，最高法院今天駁回上訴，全案定讞。最高法院指出，鍾增林於民國89年間與香港籍男子曾國財共同前往泰國購買海洛因磚，以橡膠夾藏毒品，利用海運運抵台灣，由黃姓同夥承租藏置倉庫。黃姓同夥89年5月29日將毒品運往大寮一處倉庫時，遭檢調當場查獲，最後被判處無期徒刑定讞。而鍾增林則早於同年月19日搭機出境，直到106年間因罹癌返台遭逮捕。屏東地方法院一審、台灣高等法院高雄法院二審均依運輸第一級毒品罪，判處鍾增林無期徒刑。案經上訴，最高法院認為，鍾增林坦承犯行，與多名證人指述相符，且有扣案毒品與鑑定報告等證據為證，而鍾增林事前共同謀劃犯案，參與犯行程度極深，上訴辯稱參與程度甚微，理由不足採信。最高法院認為鍾增林運輸毒品數量龐大，一旦流入市面將嚴重影響社會治安，毫無值得同情或情重法輕之處，二審判決並無違法，今天駁回上訴，全案定讞。'

from src import ClassificationModel
strategy = tf.distribute.get_strategy()


MAX_LEN = 192
with strategy.scope():
    transformer_layer = (
        transformers.TFBertModel
        .from_pretrained('bert-base-chinese')
    )
    cls = ClassificationModel.build_model(transformer_layer, max_len=MAX_LEN)

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
	art_dataset = tf.data.Dataset.from_tensor_slices(art_enc).batch(1)
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
	predict(article)
  

