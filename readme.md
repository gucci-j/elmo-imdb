IMDB classification task with ELMo
===
IMDBネガポジ分類タスクを[ELMo](https://arxiv.org/abs/1802.05365)を使ったモデルで簡単に取り組んでみました．Kerasで実装されています．

## How to Run
```
git clone git@gitlab.katfuji:gucci/elmo-imdb.git
cd elmo-imdb
python main.py 
```

* デフォルトではELMoを用いて動作するように設定されていますが，`main.py`の変数: `use_elmo`を`False`に設定することで，ELMoを用いないモデルで動作させることができます．

## Model Structure
ベースモデルとELMoを適用したモデルの二つを用意してあります．

ベースモデルの構造は以下の通りです．  
* Input -> Embedding -> 双方向LSTM -> Output  

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 80)                0
_________________________________________________________________
embedding_1 (Embedding)      (None, 80, 300)           6000000
_________________________________________________________________
bidirectional_1 (Bidirectional) (None, 256)               439296
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 257
=================================================================
Total params: 6,439,553
Trainable params: 6,439,553
Non-trainable params: 0
_________________________________________________________________
```

ELMoを適用したモデルの構造は以下の通りです．  
* Input -> Embedding + ELMo Embedding -> 双方向LSTM -> Output  

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 80)           0
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 80)           0
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 80, 300)      6000000     input_1[0][0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 80, 1024)     0           input_2[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 80, 1324)     0           embedding_1[0][0]
                                                                 lambda_1[0][0]
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 256)          1487872     concatenate_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            257         bidirectional_1[0][0]
==================================================================================================
Total params: 7,488,129
Trainable params: 7,488,129
Non-trainable params: 0
__________________________________________________________________________________________________
```

## Result
動作結果は以下の通りです．
* パラメータチューニング等はせずに雑に実験しましたが，若干のスコアの向上(約0.015)が見られました．  
有意かどうかの検証は各自でお願いします．

ELMo __なし__ のとき
```
Train on 25000 samples, validate on 25000 samples
Epoch 1/15
25000/25000 [==============================] - 79s 3ms/step - loss: 0.4453 - acc: 0.7850 - val_loss: 0.3554 - val_acc: 0.8441
Epoch 2/15
25000/25000 [==============================] - 79s 3ms/step - loss: 0.2615 - acc: 0.8951 - val_loss: 0.3706 - val_acc: 0.8392
Epoch 00002: early stopping
25000/25000 [==============================] - 24s 941us/step
Test score: 0.37064840471744537
Test accuracy: 0.8391999988555908
```

ELMo __あり__ のとき
```
Train on 25000 samples, validate on 25000 samples
Epoch 1/15
25000/25000 [==============================] - 447s 18ms/step - loss: 0.4295 - acc: 0.7986 - val_loss: 0.3244 - val_acc: 0.8578
Epoch 2/15
25000/25000 [==============================] - 446s 18ms/step - loss: 0.2636 - acc: 0.8887 - val_loss: 0.3120 - val_acc: 0.8669
Epoch 3/15
25000/25000 [==============================] - 448s 18ms/step - loss: 0.1579 - acc: 0.9392 - val_loss: 0.3871 - val_acc: 0.8552
Epoch 00003: early stopping
25000/25000 [==============================] - 205s 8ms/step
Test score: 0.3871402155160904
Test accuracy: 0.8551999998092651
```

## Note
* 事前学習済みELMoモジュールの制約上，モジュールに渡す入力データは単語分割済みの生のテキストデータである必要があります．  
そのため，main.pyでは，トークン化されているデータを生の単語列に戻す作業を行っています．ELMoモジュールを活用する際にはこの点にご注意ください．

* 今回はやっていませんが，[ベースモデルの改善](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)をすればさらに良い結果が出ると思います．

* GloVe Embedding等を使ってもよいと思います．

## Reference
[Keras imdb_lstm.py](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)  
[ELMo paper](https://arxiv.org/abs/1802.05365)

## Licence
[MIT](./LICENCE)
