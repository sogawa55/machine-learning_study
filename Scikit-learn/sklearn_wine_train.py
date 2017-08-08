#　ワインに含まれる成分を入力として、ワインのおいしさを正解ラベルに学習を行い、おいしさを判定する。

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

# ワインデータを読み込む
wine_csv = []
with open("winequality-white.csv", "r", encoding="utf-8") as fp:
    no = 0
    for line in fp:
	    # 一行づつ読み込んでいく
	    line = line.strip()
		# セミコロンで行を区切って格納していく
	    cols = line.split(";")
	    wine_csv.append(cols)

# 1行目はヘッダ行なので削除
wine_csv = wine_csv[1:]

# CSVの各データを数値に変換する
labels = []
data = []
for cols in wine_csv:
    # map関数でリストの各要素を数値に変換
    cols = list(map(lambda n: float(n), cols))
    # ワインの評価数値を格納
    grade = int(cols[11])
	# 評価数値の調整
    if grade == 9: grade = 8
    if grade < 4: grade = 5
	# 正解ラベルとしてワインの評価数値を格納
    labels.append(grade)
	# 成分値を配列として格納
    data.append( cols[0:11])

# 訓練用データとテストデータに分ける
data_train, data_test, label_train, label_test = train_test_split(data, labels)

# ランダムフォレストのアルゴリズムを使用してモデルを作成
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(data_train, label_train)
# SVMのアルゴリズムを使用したモデルの作成
#clf = svm.SVC()
#clf.fit(data_train, label_train)

# 予測の実行
predict = clf.predict(data_test)
# 正解データをもとに正答率を算出
ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("正解率=", ac_score)
print("レポート=\n",cl_report)