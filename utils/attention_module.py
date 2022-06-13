import tensorflow as tf
from utils.brunch import Mask_Brunch, Trunk_Brunch
from utils.residual_unit import Residual_Unit

class Attention_Module(tf.keras.layers.Layer):
    def __init__(self, channels, stage, p=1, t=2, r=1, **kwargs):
        super(Attention_Module, self).__init__(**kwargs)
        self.channels = channels
        self.stage = stage
        self.p = p
        self.t = t
        self.r = r
        for i in range(2*self.p):
            setattr(self, f'residual_units{i}', Residual_Unit(self.channels))
            
        self.mask_brunch = Mask_Brunch(self.channels, r=self.r, stage=self.stage)
        self.trunk_brunch = Trunk_Brunch(self.channels, t=self.t)
        self.mul = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()
        
    def call(self, x):
        for i in range(self.p):
            x = getattr(self, f'residual_units{i}')(x)
            
        M = self.mask_brunch(x)
        T = self.trunk_brunch(x)
        x = self.mul([M, T])
        x = self.add([x, T])
        
        for j in range(self.p):
            x = getattr(self, f'residual_units{i+j+1}')(x)
        return x