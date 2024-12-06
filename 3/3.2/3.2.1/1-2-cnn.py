#（2）深度学习方法
#使用 CNN 进行静态表情识别的 Python 代码如下。
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.utils import load_img, img_to_array
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import model_from_json
# 加载数据集（请替换为你的数据集路径）
data = np.load("your_dataset.npz")
X = data["X"]
y = data["y"]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 数据预处理
X_train = np.array([img_to_array(load_img(img, target_size=(48, 48))) for img in X_train])
X_test = np.array([img_to_array(load_img(img, target_size=(48, 48))) for img in X_test])
# 转换为张量并归一化
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0
#将标签转换为独热编码
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 构建CNN模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(<num_classes>, activation='softmax')) # num_classes为类别数量
# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
# 保存模型
model.save("cnn_model.h5")
# 加载模型进行预测
loaded_model = model_from_json(open("cnn_model.json", "r").read())
loaded_model.load_weights("cnn_model.h5")
predictions = loaded_model.predict(X_test)
# 输出预测结果
print(predictions)