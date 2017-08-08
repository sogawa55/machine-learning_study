# 日経平均株価の過去の推移から、株価を予測する
# 訓練用データと正解ラベル(一日後の終値と始値の差)を元に教師あり学習を行う
# テストデータと正解ラベルと突き合わせて、正解率を検証する

import numpy as np
import pandas as pd
import os
from numpy.random import *
from scipy.stats import zscore

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.optimizers import RMSprop

# 学習モデルの作成
def BuildNeuralNetwork(input):
    # 多層のSequentialモデルを実行
    model = Sequential()
	# 最初の層で入力の形を指定する
	# 128次元の隠れ層を実装
    model.add(Dense(128, input_shape=(input,)))
	#活性化関数(Relu関数)を実装
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
	#Dropoutで過学習対策
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
	#1次元の出力層に変換
    model.add(Dense(1))
	#出力層で線形回帰を実装
    model.add(Activation('linear'))
    model.compile(loss="mean_squared_error", optimizer=RMSprop())
    return model

#訓練用データとテスト用データに分割
def MakeDataSet(nd_data):
    # 全体の20%をランダムにテストデータにする
    test_rate = int(len(nd_data)*0.2)
	#　randint関数でランダム化
    test_index = np.unique(randint(0,len(nd_data),test_rate))
    #　astypeメソッドでint型に変換
    test_index = test_index.astype(np.int64)
    train_index = np.array([])
    for i in range(len(nd_data)):
	　　　　if(len(np.where(test_index == i)[0]) == 0):
            train_index = np.append(train_index,i)
    train_index = train_index.astype(np.int64)
    return nd_data[train_index,0:4],nd_data[train_index,4:5],nd_data[test_index,0:4],nd_data[test_index,4:5]

#　学習を定義
def trainModel(model,train,answer,batchs,epochs):
    TensorBoard_cb = TensorBoard(log_dir="./TensorBoard_log/", histogram_freq=1)
    hist = model.fit(train, answer,batch_size=batchs,verbose=1,epochs=epochs,callbacks=[TensorBoard_cb])
    return hist

#CSVファイルの読み込みメソッドを定義
def GetDataFrame(DIR):
    #pandasでデータフレームを作成
    data = pd.DataFrame()
	#osのlistdirメソッドで指定のpathを読み込む
    csvs = [DIR+i for i in os.listdir(DIR)] # use this for training images
    csvs.reverse()
    for csv in csvs:
	　　　　#.DS_Store対策
        if csv.find('.DS_Store') == -1:
            tmp = pd.read_csv(csv)
            data = data.append(tmp,ignore_index = True)
    return data

#学習に用いるバッチ数
BATCHES = 10
#訓練データセットの学習回数
EPOCHS = 1000

path = r'C:/Users/shohei/nikkei/'
data =GetDataFrame(path)
#終値から始値を引き、変化を格納
data['diff'] = data['end'] - data['start'] 
data['predict_reg'] = 0
#１日後のdiffを正解データにする
data['predict_reg'][1:]= data['diff'][0:-1] 
#0番目のデータは正解データがNULLになるので不採用
use_data = data[1:] 

# 学習データは平均0、分散1になる様に標準化
# pandasのlocメソッドで行ラベルと列ラベルを取得
# scipyのzscoreメソッドで標準化を行う。(Zスコア=平均0,分散1)
use_data.loc[:,'start'] = zscore(use_data['start'])
use_data.loc[:,'max'] = zscore(use_data['max'])
use_data.loc[:,'min'] = zscore(use_data['min'])
use_data.loc[:,'end'] = zscore(use_data['end'])

# 正解データのみ最大値を用いて-1〜1の範囲に標準化
use_data.loc[:,'predict_reg'] /= max(use_data['predict_reg'])

# as_matrixメソッドを用いてPandas.DataFrameをnumpyのarray型の多次元配列に変換する
nd_data = use_data[['start','max','min','end','predict_reg']].as_matrix()
# 上記でnumpy型に変換したデータを訓練用データとテストデータに分ける。
x_train,x_test,y_train,y_test = MakeDataSet(nd_data)

# モデルをインスタンス化
model = BuildNeuralNetwork(4)
# 訓練用データをもとに学習を実行
trainModel(model,x_train,x_test,BATCHES,EPOCHS)
# テストデータを基に予測を実行
predicted = model.predict(y_train)
prediction_ratio = 0
# 正解データ(一日後の終値と始値の差)をベースに予測の答え合わせを行う
# 正解データと符号が一致しているかを一致率で評価
for i in range(len(y_test)):
　　　　#予測データが正の値の場合
    if(predicted[i] >= 0):
        if (y_test[i] > 0):
            prediction_ratio += 1　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
        print("next day will increase by ",predicted[i]," --------",y_test[i])#正解データとの突合せ
　　　　#予測データが負の値の場合
    else:
        if (y_test[i] <= 0):
            prediction_ratio += 1
        print("next day will decrease by ",predicted[i]," --------",y_test[i])#正解データとの突合せ
print("Ratio is",((prediction_ratio/len(y_test)) * 100),"%")
K.clear_session()