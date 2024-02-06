# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import torch
import tensorflow as tf
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi，python, {name}')  # 按 Ctrl+F8 切换断点。


# # 按装订区域中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     print_hi('PyCharm')
#     tensor = torch.ones(size=(2, 3, 4, 5))
#     print(tensor)
#
#     tf.config.list_physical_devices('GPU')


# print(torch.cuda.get_device_name(0))
# print(tf.config.list_physical_devices('GPU'))


# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test, verbose=2)

