import numpy as np 
from keras.models import Sequential,model_from_yaml,load_model
from keras.layers import Dense,Dropout,LSTM
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.utils import np_utils

raw_text=open('E:/dataset/NLP/adult/keras_to_seq.txt').read()
raw_text=raw_text.lower()

chars=sorted(list(set(raw_text)))
char_to_int=dict((c,i) for i,c in enumerate(chars))
int_to_char=dict((i,c) for i,c in enumerate(chars))

seq_length=100
x=[]
y=[]
for i in range(0,len(raw_text)-seq_length):
    given=raw_text[i:i+seq_length]
    predict=raw_text[i+seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

print(x[:3])
print(y[:3])

n_patterns=len(x)
n_vocab=len(chars)

# change x into the need of lstm
x=np.reshape(x,(n_patterns,seq_length,1))
# normalization (0,1)
x=x/float(n_vocab)
# change output into one-hot
y=np_utils.to_categorical(y)

print(x[11])
print(y[11])
'''
model=Sequential()
model.add(LSTM(128,input_shape=(x.shape[1],x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')


print("train================================")
best_model=ModelCheckpoint("E:/ML/models/keras_do_seq.h5",monitor='loss',verbose=1,save_best_only=True,mode='min')
tbcallbacks=TensorBoard(log_dir='E:/ML/.vscode/dc_logs/keras_do_seq',histogram_freq=0,write_graph=True)
model.fit(x,y,epochs=10,batch_size=32,callbacks=[best_model,tbcallbacks])

yaml_string=model.to_yaml()
with open('E:/ML/models/keras_do_seq.yaml','w') as outfile:
    outfile.write(yaml_string)
'''
print("predict================================")
with open('E:/ML/models/keras_do_seq.yaml') as yamlfile:
    loaded_model_yaml=yamlfile.read()

model=model_from_yaml(loaded_model_yaml)
model.load_weights('E:/ML/models/keras_do_seq.h5')

def string_to_index(raw_input):   #将生词 转换成 int 数组
    res = []
    for e in raw_input[(len(raw_input)-seq_length):]:
        res.append(char_to_int[e])
    return res

def predict_next(input_array): #预测下一个y
    x=np.reshape(input_array,(1,seq_length,1))
    # x = np.reshape(input_array,(1,seq_length,1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y

def y_to_char(y):                   # 将y输出 字符e
    largest_index = y.argmax()
    e = int_to_char[largest_index]
    return e

def generate_article(init,rounds =50):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))        
        in_string += n
    return in_string

init='Before I went to middle school, I lived with my grandparents. At that time, I stayed in the country, which was the most beautiful place for me. Early in the morning, I walked along the country road, appreciating the fresh air. We raised a little dog. He was my best friend, and he accompanied me all the time. No matter what I did and where I went, the little dog was with me and I loved him so much. Sometimes I would pick up the fruit in the garden, sometimes I would catch the fish in the clear river. '
article=generate_article(init)
print(article)