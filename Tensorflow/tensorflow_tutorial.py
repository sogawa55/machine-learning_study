
#Tensorflow公式チュートリアル(MNIST For ML Beginners) 
#手書き文字の多クラス分類

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # 学習用、テスト用、検証用のデータ読み込む
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  
  # 入力xを格納するプレースホルダーを定義
  x = tf.placeholder(tf.float32, [None, 784])
  # 重み・バイアスの設定　
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  # 入力xに対して、0-9の分類を行うモデルを実装
  y = tf.matmul(x, W) + b

  # 交差エントロピー(正解との誤差)を測定するためのプレースホルダー
  y_ = tf.placeholder(tf.float32, [None, 10])

  #モデルと正解値を引数に交差エントロピーを実装する
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  #勾配降下アルゴリズムを使用して学習率0.5で交差エントロピーを最小化させる
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  # 演算実行
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # 訓練ステップを1000回実行
  for _ in range(1000):
    #訓練セットから100個のランダムなデータポイントのバッチを取得する
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # 正解のラベルとどれだけ等しいか、モデルの学習結果を評価する
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # テストで得たbool型の値を浮動小数型にキャストし、reduce_meanメソッドで平均を取る
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # 画像入力と正解ラベルを引数に、accuracyを計算して精度を求める
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)