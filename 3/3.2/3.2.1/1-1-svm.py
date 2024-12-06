#（1）支持向量机
#使用SVM进行静态表情识别的Python代码如下
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
# 加载数据集（请替换为你的数据集路径）
data = np.load("your_dataset.npz")
X = data["X"]
y = data["y"]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 数据预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# SVM模型训练
svm_model = SVC(kernel="rbf", C=1, gamma="scale")
svm_model.fit(X_train, y_train)
# 预测
y_pred = svm_model.predict(X_test)
# 输出分类报告
print(classification_report(y_test, y_pred))