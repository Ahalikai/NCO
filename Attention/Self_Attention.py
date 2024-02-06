from tensorflow.keras.preprocessing import sequence
# 数据集导入
from keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# 自定义layers所需库
from keras import backend as K
# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer


from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.layers import *

class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重,名为kernel，权重初始化为uniform(均匀分布)，
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        print("attention input:", input_shape[0], input_shape[1], input_shape[2])
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)
        # 数据转置 K.permute_dimensions(X,(x,y,z))
        # 对应numpy就是np.transpose
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        # scaled
        QK = QK / (self.output_dim ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)
        # 矩阵叉乘
        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

if __name__ == '__main__':
    max_features = 2000

    print('Loading data...')
    # num_words 大于该词频的单词会被读取,如果单词的词频小于该整数，会用oov_char定义的数字代替。默认是用2代替。
    # 需要注意的是，词频越高的单词，其在单词表中的位置越靠前，也就是索引越小，所以实际上保留下来的是索引小于此数值的单词
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print('Loading data finish!')

    # 标签转换为独热码
    # get_dummies是利用pandas实现one-hot encode的方式。
    y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test)
    print(x_train.shape, 'train sequences')
    print(x_test.shape, 'test sequences')

    maxlen = 128
    batch_size = 32

    # 将序列转化为经过填充以后的一个长度相同的新序列新序列，默认从起始位置补，当需要截断序列时，从起始截断，默认填充值为0
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print('x_train shape:', x_train.shape)

    print('x_test shape:', x_test.shape)
    '''
    output:
    (25000,) train sequences
    (25000,) test sequences
    x_train shape: (25000, 128)
    x_test shape: (25000, 128)
    '''


    S_inputs = Input(shape=(maxlen,), dtype='int32')

    # 词汇表大小为max_features(2000),词向量大小为128，需要embedding词的大小为S_inputs
    embeddings = Embedding(max_features, 128)(S_inputs)

    O_seq = Self_Attention(128)(embeddings)

    # GAP 对时序数据序列进行平均池化操作，也可以采用GMP(Global Max Pooling)
    O_seq = GlobalAveragePooling1D()(O_seq)

    O_seq = Dropout(0.5)(O_seq)

    # 输出为二分类
    outputs = Dense(2, activation='softmax')(O_seq)

    model = Model(inputs=S_inputs, outputs=outputs)

    print(model.summary())
    # 优化器选用adam
    opt = Adam(lr=0.0002, decay=0.00001)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    print('Train start......')

    h = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=5,
                  validation_data=(x_test, y_test))

    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.plot(h.history["accuracy"], label="train_acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.show()







