"""
@Time: 2025/4/29 11:13
@Author: yanzx
@Desc: 
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
#
# # 1. 模拟数据
# num_users = 1000
# num_items = 5000
# num_samples = 10000
#
# # 随机生成用户ID、物品ID和标签（0/1表示是否交互）
# user_ids = np.random.randint(0, num_users, size=num_samples)
# item_ids = np.random.randint(0, num_items, size=num_samples)
# labels = np.random.randint(0, 2, size=num_samples)  # 二分类标签
#
# # 2. 构建双塔模型
# def build_two_tower_model():
#     # 用户塔
#     user_input = Input(shape=(1,), name="user_input")
#     user_embedding = Embedding(num_users, 32)(user_input)
#     user_flatten = Flatten()(user_embedding)
#     user_dense = Dense(64, activation="relu")(user_flatten)
#
#     # 物品塔
#     item_input = Input(shape=(1,), name="item_input")
#     item_embedding = Embedding(num_items, 32)(item_input)
#     item_flatten = Flatten()(item_embedding)
#     item_dense = Dense(64, activation="relu")(item_flatten)
#
#     # 计算点积相似度
#     dot_product = tf.keras.layers.Dot(axes=1)([user_dense, item_dense])
#     output = Dense(1, activation="sigmoid")(dot_product)  # 二分类输出
#
#     model = Model(inputs=[user_input, item_input], outputs=output)
#     return model
#
# model = build_two_tower_model()
# model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
#
# # 3. 训练
# model.fit(
#     [user_ids, item_ids], labels,
#     batch_size=64,
#     epochs=10,
#     validation_split=0.2
# )
#
# # 4. 预测
# test_user = np.array([1, 2, 3])  # 用户ID
# test_item = np.array([10, 20, 30])  # 物品ID
# predictions = model.predict([test_user, test_item])
# print("预测得分:", predictions.flatten())