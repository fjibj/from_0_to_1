#1．基于GAN的表情生成
#CGAN Python 代码示例如下。
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, LeakyReLU
from tensorflow.python.keras.layers import BatchNormalization, Activation, Embedding, multiply
from tensorflow.python.keras.layers import Conv2DTranspose, Conv2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
import numpy as np

# 定义生成器模型
def build_generator(z_dim, num_classes, img_shape):
 noise_input = Input(shape=(z_dim, ))
 label_input = Input(shape=(1, ), dtype='int32')
 label_embedding = Embedding(num_classes, z_dim)(label_input)
 label_embedding = Flatten()(label_embedding)
 joined_representation = multiply([noise_input, label_embedding])
 generator = Dense(256, input_dim=z_dim*num_classes)(joined_representation)
 generator = LeakyReLU(alpha=0.2)(generator)
 generator = BatchNormalization(momentum=0.8)(generator)
 generator = Dense(512)(generator)
 generator = LeakyReLU(alpha=0.2)(generator)
 generator = BatchNormalization(momentum=0.8)(generator)
 generator = Dense(1024)(generator)
 generator = LeakyReLU(alpha=0.2)(generator)
 generator = BatchNormalization(momentum=0.8)(generator)
 generator = Dense(np.prod(img_shape), activation='tanh')(generator)
 generator = Reshape(img_shape)(generator)
 gen_model = Model(inputs=[noise_input, label_input], outputs=[generator])
 return gen_model

# 定义判别器模型
def build_discriminator(img_shape, num_classes):
 img_input = Input(shape=img_shape)
 label_input = Input(shape=(1, ), dtype='int32')
 label_embedding = Embedding(num_classes, np.prod(img_shape))(label_input)
 label_embedding = Flatten()(label_embedding)
 flat_img = Flatten()(img_input)
 merged_input = Concatenate([flat_img, label_embedding])
 discriminator = Dense(1024)(merged_input)
 discriminator = LeakyReLU(alpha=0.2)(discriminator)
 discriminator = Dense(512)(discriminator)
 discriminator = LeakyReLU(alpha=0.2)(discriminator)
 discriminator = Dense(256)(discriminator)
 discriminator = LeakyReLU(alpha=0.2)(discriminator)
 discriminator = Dense(1, activation='sigmoid')(discriminator)
 disc_model = Model(inputs=[img_input, label_input], outputs=[discriminator])
 return disc_model

# 定义CGAN模型
def build_cgan(generator, discriminator):
 z_dim = generator.input_shape[0][1]
 num_classes = discriminator.input_shape[1][1]
 noise_input = generator.input[0]
 label_input = generator.input[1]
 img = generator([noise_input, label_input])
 discriminator.trainable = False
 valid = discriminator([img, label_input])
 cgan = Model([noise_input, label_input], valid)
 cgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
 return cgan

# 定义训练函数
def train_cgan(generator, discriminator, cgan, X_train, y_train, z_dim, num_classes,
epochs, batch_size):
 valid = np.ones((batch_size, 1))
 fake = np.zeros((batch_size, 1))
 for epoch in range(epochs):
  for _ in range(X_train.shape[0] // batch_size):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    labels = y_train[idx]
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    gen_images = generator.predict([noise, labels])
    d_loss_real = discriminator.train_on_batch([real_images, labels], valid)
    d_loss_fake = discriminator.train_on_batch([gen_images, labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    valid_labels = np.ones((batch_size, 1))
    g_loss = cgan.train_on_batch([noise, labels], valid_labels)
  print(f"Epoch {epoch}, D Loss : {d_loss[0]}, G Loss : {g_loss}")
 return generator

# 设置参数
z_dim = 100 # 噪声向量维度
num_classes = N # 类别数量
img_shape = (64, 64, 3) # 图像形状
epochs = 10000
batch_size = 64
# 创建并编译生成器和判别器
generator = build_generator(z_dim, num_classes, img_shape)
discriminator = build_discriminator(img_shape, num_classes)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5),
metrics=['accuracy'])
# 创建并编译CGAN模型
cgan = build_cgan(generator, discriminator)
# 加载和准备数据（请根据实际情况更改）
X_train = ...
y_train = ...
# 训练CGAN模型
trained_generator = train_cgan(generator, discriminator, cgan, X_train, y_train,
              z_dim, num_classes, epochs, batch_size)
# 生成表情图像的示例
noise = np.random.normal(0, 1, (1, z_dim))
label = np.array([0]) # 替换为所需的标签
generated_image = trained_generator.predict([noise, label])