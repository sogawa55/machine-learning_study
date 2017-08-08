#　手書き数字の認識

from sklearn import datasets,svm,metrics
#　sklearn0.20に対応
from sklearn.model_selection import train_test_split
# 手書き数字データを読み込む
digits = datasets.load_digits()
# 訓練用データとテストデータに分ける
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target)

# SVMアルゴリズムを用いてモデルを構築する
# gammaはハイパーパラメータ、謝分類に対するペナルティの大きさ
clf = svm.SVC(gamma=0.001)
# 学習の実行
clf.fit(data_train, label_train)
# 予測の実行
predict = clf.predict(data_test)
# 正解データをもとに正答率を算出
ac_score = metrics.accuracy_score(label_test, predict)
c1_report = metrics.classification_report(label_test, predict)
print("分類器の情報=", clf)
print("正解率=", ac_score)
print("レポート=\n", c1_report)