from pandas import read_csv,DataFrame,concat
from datetime import datetime
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from keras.models import Sequential,load_model,model_from_yaml
from keras.layers import LSTM,Dense
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df=DataFrame(data)
    cols,names=list(),list()

    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)'%(j+1,i)) for j in range(n_vars)]
    # print("1========================")
    # print(names[::])
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('var%d(t)'%(j+1)) for j in range(n_vars)]
        else:
            names+=[('var%d(t+%d)'%(j+1,i)) for j in range(n_vars)]
    # print("2========================")
    # print(names[::])
    agg=concat(cols,axis=1)
    agg.columns=names
    # print("3========================")
    # print(agg[::])
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def load_data():
    def parse(x):
        return datetime.strptime(x,'%Y %m %d %H')
    dataset=read_csv('E:/dataset/regression/raw.csv',parse_dates=[['year','month','day','hour']],index_col=0,date_parser=parse)
    dataset.drop('No',axis=1,inplace=True)
    dataset.columns=['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
    dataset.index.name='date'
    dataset['pollution'].fillna(0,inplace=True)
    dataset=dataset[24:]
    print(dataset.head(5))
    dataset.to_csv('E:/dataset/regression/pollution.csv')

    dataset=read_csv('E:/dataset/regression/pollution.csv',header=0,index_col=0)
    values=dataset.values
    encoder=LabelEncoder()
    values[:,4]=encoder.fit_transform(values[:,4])
    values=values.astype('float32')
    test1=values[9999][0:8]
    test1_y=values[10000]
    test_test1=np.asarray([[test1]])

    # print("---------------")
    # print(values.shape)
    x_min=np.min(values,axis=0)
    x_max=np.max(values,axis=0)
    # print(x_min,x_max)
    # print(x_min[0:1],x_max[0:1])

    # print("====================")
    # print(test_test1.shape)
    # print(test1,test1_y)

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled=scaler.fit_transform(values)
    reframed=series_to_supervised(scaled,1,1)
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]],axis=1,inplace=True)
    print(reframed.head())

    values=reframed.values
    n_train_hours=365*24
    train=values[:n_train_hours,:]
    test=values[n_train_hours:,:]

    train_X,train_y=train[:,:-1],train[:,-1]
    test_X,test_y=test[:,:-1],test[:,-1]

    train_X=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    test_X=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)

    predict_test=test_X[9999:10000:]
    predict_test_y=test_y[9999:10000]
    print(predict_test)
    print(predict_test_y)

    return train_X,train_y,test_X,test_y,predict_test,predict_test,x_max,x_min

# def figure():
#     dataset=read_csv('E:/dataset/regression/pollution.csv',header=0,index_col=0)
#     values=dataset.values
#     groups=[0,1,2,3,5,6,7]
#     i=1
#     plt.figure()
#     for group in groups:
#         plt.subplot(len(groups),1,i)
#         plt.plot(values[:,group])
#         plt.title(dataset.columns[group],y=0.5,loc='right')
#         i+=1
#     plt.show()


def train_model():

    train_X,train_y,test_X,test_y,predict_test,predict_test_y,x_max,x_min=load_data()
    
    model=Sequential()
    model.add(LSTM(50,input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    
    history=model.fit(train_X,train_y,epochs=50,batch_size=128,validation_data=(test_X,test_y),verbose=2,shuffle=False)

    yaml_string=model.to_yaml()
    with open('E:/ML/models/pollution.yaml','w') as outfile:
        outfile.write(yaml_string)
    model.save_weights('E:/ML/models/pollution.h5')

    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='test')
    plt.legend()
    plt.show()

def predict_data():
    train_X,train_y,test_X,test_y,predict_test,predict_test_y,x_max,x_min=load_data()
    with open('E:/ML/models/pollution.yaml') as yamlfile:
        loaded_model_yaml=yamlfile.read()
    model=model_from_yaml(loaded_model_yaml)
    model.load_weights('E:/ML/models/pollution.h5')
    test_predicted=model.predict(predict_test)
    #invert to the actual data is wrong
    test_true=test_predicted*(x_max[0:1]-x_min[0:1])+x_min[0:1]
    print(model.predict(predict_test))
    print(test_true)


if __name__=='__main__':
    
    # train_model()
    predict_data()