import tensorflow as tf

class Residual_Unit(tf.keras.layers.Layer):
    def __init__(self, channels, strides=1, **kwargs):
        super(Residual_Unit, self).__init__(**kwargs)
        self.channels = channels
        self.strides = strides
        
        self.conv1 = tf.keras.layers.Conv2D(filters=self.channels//4, kernel_size=1, strides=self.strides, padding='valid', use_bias=False)
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        
        self.conv2 = tf.keras.layers.Conv2D(filters=self.channels//4, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        
        self.conv3 = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.BN3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        if self.strides != 1:
            self.conv1a = tf.keras.layers.Conv2D(
                filters=self.channels, 
                kernel_size=1, 
                strides=self.strides, 
                padding='valid', 
                use_bias=False
            )
        
    def call(self, x):
        if self.strides != 1:
            shortcut = self.conv1a(x)
        else:
            shortcut = x
        outputs = self.conv1(x)
        outputs = self.BN1(outputs)
        outputs = self.relu1(outputs)
        
        outputs = self.conv2(outputs)
        outputs = self.BN2(outputs)
        outputs = self.relu2(outputs)
        
        outputs = self.conv3(outputs)
        outputs = self.BN3(outputs)
            
        outputs = tf.add(outputs, shortcut)
        outputs = self.relu3(outputs)
        return outputs