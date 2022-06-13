import tensorflow as tf
from utils.residual_unit import Residual_Unit

class Trunk_Brunch(tf.keras.layers.Layer):
    def __init__(self, channels, t=2, **kwargs):
        super(Trunk_Brunch, self).__init__(**kwargs)
        self.channels = channels
        self.t = t
        self.residual_units = []
        for i in range(self.t):
            setattr(self, f'residual_units{i}', Residual_Unit(self.channels))
            
    def call(self, x):
        for i in range(self.t):
            x = getattr(self, f'residual_units{i}')(x)
        return x

class Mask_Brunch(tf.keras.layers.Layer):
    def __init__(self, channels, r=1, stage=0, **kwargs):
        super(Mask_Brunch, self).__init__(**kwargs)
        self.r = r
        self.channels = channels
        self.num_of_pool = 3- stage

        for i in range(self.num_of_pool):
            setattr(self, f'maxpool{i}', tf.keras.layers.MaxPool2D((2, 2)))
            for j in range(self.r):
                setattr(self, f'residual_units{i}_{j}', Residual_Unit(self.channels))
            setattr(self, f'skip_residual_units{i}', Residual_Unit(self.channels))

        self.maxpooling = tf.keras.layers.MaxPool2D((2, 2))
        for k in range(2 * self.r):
            setattr(self, f'nested_residual_units{k}', Residual_Unit(self.channels))
        self.interpolation = tf.keras.layers.UpSampling2D(size=(2, 2))

        for l in range(self.num_of_pool):
            setattr(self, f'add{i}', tf.keras.layers.Add())
            for m in range(self.r):
                setattr(self, f'residual_units{l+i+1}_{m}', Residual_Unit(self.channels))
            setattr(self, f'interpolation{l}', tf.keras.layers.UpSampling2D(size=(2, 2)))
            
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, use_bias=False)
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        
        self.conv2 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
    def call(self, x):
        for i in range(self.num_of_pool):
            x = getattr(self, f'maxpool{i}')(x)
            for j in range(self.r):
                x = getattr(self, f'residual_units{i}_{j}')(x)
            setattr(self, f'skip_residual{i}', getattr(self, f'skip_residual_units{i}')(x))

        x = self.maxpooling(x)
        for k in range(2 * self.r):
            x = getattr(self, f'nested_residual_units{k}')(x)
        x = self.interpolation(x)
        
        for l in range(self.num_of_pool):
            x = getattr(self, f'add{i}')([x, getattr(self, f'skip_residual{self.num_of_pool-l-1}')])
            for m in range(self.r):
                x = getattr(self, f'residual_units{l+i+1}_{m}')(x)
            x = getattr(self, f'interpolation{l}')(x)
            
        x = self.BN1(x)
        x = self.relu1(x)
        
        x = self.conv1(x)
        x = self.BN2(x)
        x = self.relu2(x)
        
        x = self.conv2(x)
        return self.sigmoid(x)