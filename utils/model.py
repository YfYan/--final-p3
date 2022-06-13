import tensorflow as tf

from utils.attention_module import Attention_Module
from utils.residual_unit import Residual_Unit


class ResidualAttentionNetwork(tf.keras.Model):
    def __init__(self, p=1, t=2, r=1, **kwargs):
        super(ResidualAttentionNetwork, self).__init__(**kwargs)
        self.p = p
        self.t = t
        self.r = r
        
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv1_BN = tf.keras.layers.BatchNormalization()
        self.conv1_activation = tf.keras.layers.ReLU()
        
        self.residual_unit1 = Residual_Unit(channels=128, strides=2)
        self.attention_module1 = Attention_Module(channels=128, stage=2, p=1, t=2, r=1)
        
        self.residual_unit2 = Residual_Unit(channels=256, strides=2)
        self.attention_module2 = Attention_Module(channels=256, stage=3, p=1, t=2, r=1)
        
        self.residual_unit3 = Residual_Unit(channels=512, strides=2)
        self.residual_unit4 = Residual_Unit(channels=512)
        self.residual_unit5 = Residual_Unit(channels=512)
        
        self.BN = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=4, strides=1)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=100,activation = 'softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv1_BN(x)
        x = self.conv1_activation(x)

        x = self.residual_unit1(x)
        x = self.attention_module1(x)

        x = self.residual_unit2(x)
        x = self.attention_module2(x)

        x = self.residual_unit3(x)
        x = self.residual_unit4(x)
        x = self.residual_unit5(x)

        x = self.BN(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.dense(x)