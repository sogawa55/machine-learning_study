#　手書き数字画像の認識

import os, sys, math
from sklearn import datasets, svm
from sklearn.externals import joblib

DIGITS_PKL = "digit-clf.pk1"

# 予測モデルを作成する
def train_digits():
    digits = datasets.load_digits()
	# 学習用データ
    data_train = digits.data
	# 学習用ラベル
    label_train = digits.target
	# SVMアルゴリズムでモデルを作成
    clf = svm.SVC(gamma=0.001)
	# 学習開始
    clf.fit(data_train, label_train)
	# 予測モデルを保存
    joblib.dump(clf, DIGITS_PKL)
    print("予測モデルを保存しました=", DIGITS_PKL)
    return clf

# データから数字を予測する
def predict_digits(data):
    # モデルが存在しない場合に実行
    if not os.path.exists(DIGITS_PKL):
	    clf = train_digits()
	# モデルの読み込みを行う
    clf = joblib.load(DIGITS_PKL)
	# 予測を実行
    n = clf.predict([data])
    print("判定結果=", n)

# 手書き数字画像を8×8グレイスケールのデータ配列に変換
def image_to_data(imagefile):
    import numpy as np
    from PIL import Image
	# convertメソッドで画像をグレイスケールに変換する
    image = Image.open(imagefile).convert('L')
	# resizeメソッドを使って8×8ピクセルにリサイズ(アンチエイリアス処理も行う)
    image = image.resize((8,8), Image.ANTIALIAS)
	# numpy.ndarray型の多次元配列に変換する
    img = np.asarray(image, dtype=float)
	#　ndarray配列の全要素=全ピクセルに対して計算を行う
	# floorは四捨五入
    img = np.floor(16-16 * (img/256))
    img = img.flatten()
    print(img)
    return img

def main():
    # コマンドライン引数を得る
    if len(sys.argv) <= 1 :
	    print("USAGE:")
	    print("python3 predict_digit.py imagefile")
	    return
    imagefile = sys.argv[1]
    data = image_to_data(imagefile)
    predict_digits(data)
	
if __name__ == '__main__':
    main()
