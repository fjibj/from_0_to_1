#2．基于编码器-解码器的表情生成
#VAE Python 代码示例如下。
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
# 定义VAE模型
def build_vae(input_dim, latent_dim):
 # 编码器
 input_img = Input(shape=(input_dim, ))
 encoder = Dense(256, activation='relu')(input_img)
 z_mean = Dense(latent_dim)(encoder)
 z_log_var = Dense(latent_dim)(encoder)

 # 采样层
 def sampling(args):
 z_mean, z_log_var = args
 epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
 return z_mean + K.exp(0.5 * z_log_var) * epsilon

 z = Lambda(sampling)([z_mean, z_log_var])

 # 解码器
 decoder_input = Input(shape=(latent_dim, ))
 decoder = Dense(256, activation='relu')(decoder_input)
 output_img = Dense(input_dim, activation='sigmoid')(decoder)

 # 构建编码器和解码器
 encoder_model = Model(input_img, [z_mean, z_log_var, z])
 decoder_model = Model(decoder_input, output_img)

 # 构建VAE模型
 output_img = decoder_model(encoder_model(input_img)[2])
 vae = Model(input_img, output_img)

 # 定义VAE的损失函数
 reconstruction_loss = tf.keras.losses.binary_crossentropy(input_img, output_img)
 reconstruction_loss *= input_dim
 kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
 kl_loss = K.sum(kl_loss, axis=-1)
 kl_loss *= -0.5
 vae_loss = K.mean(reconstruction_loss + kl_loss)
 vae.add_loss(vae_loss)

 return vae, encoder_model, decoder_model

# 设置参数
input_dim = 64 * 64 * 3 # 图像数据维度
latent_dim = 100 # 潜在空间维度
# 创建并编译VAE模型
vae, encoder, decoder = build_vae(input_dim, latent_dim)
vae.compile(optimizer='adam')
# 加载和准备数据（请根据实际情况更改）
X_train = ...
# 训练VAE模型
vae.fit(X_train, epochs=epochs, batch_size=batch_size)
# 使用VAE生成表情图像的示例
z_sample = np.random.normal(0, 1, (1, latent_dim))
generated_image = decoder.predict(z_sample)