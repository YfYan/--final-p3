# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:03:18 2022

@author: yfyan
"""

import tensorflow as tf
from tensorflow.keras import layers,optimizers,datasets,Sequential
import os
import numpy as np
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from utils.model import ResidualAttentionNetwork
import config

def preprocess(x,y):
    x=2 * tf.cast(x,dtype=tf.float32)/255.-1
    y=tf.cast(y,dtype=tf.int32)
    return x,y

(x,y),(x_test,y_test) = datasets.cifar100.load_data()
x = x.astype(np.float32)
x_test = x_test.astype(np.float32)
y=tf.squeeze(y,axis=1)
y_test=tf.squeeze(y_test,axis=1)

train_db=tf.data.Dataset.from_tensor_slices((x,y))
train_db=train_db.map(preprocess).batch(50)

test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db=test_db.map(preprocess).batch(50)

MODEL_NAME = config.MODEL_NAME
p = config.p 
t = config.t 
r = config.r 

resattnet = ResidualAttentionNetwork(p=p, t=t, r=r, name=MODEL_NAME)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_baseline'
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

optimizer=optimizers.Adam(lr=1e-4)
model = resattnet
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.Mean('train_accuracy', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.Mean('test_accuracy', dtype=tf.float32)

for epoch in range(50):
    # if epoch % 5 == 4:
    #     lr/=10
    #     optimizer.lr = lr
    for step,(x,y) in enumerate(train_db):
        #这里做一个前向循环,将需要求解梯度放进来
        with tf.GradientTape() as tape:
            y_onehot=tf.one_hot(y,depth=100)
            # x, y_onehot = cutmix_mask(x,y_onehot)
            #[b,32,32,3] => [b,100]
            logits=model(x)
            #[b] => [b,100]
            #compute loss
            loss=tf.losses.categorical_crossentropy(y_onehot,logits)
    
            loss=tf.reduce_mean(loss)
            
        pred=tf.argmax(logits,axis=1)
        pred=tf.cast(pred,dtype=tf.int32)
        correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct=tf.reduce_sum(correct)
        acc = correct/x.shape[0]
        
        train_loss(loss)
        train_accuracy(acc)
        #计算gradient
        grads=tape.gradient(loss,model.trainable_variables)
        #传给优化器两个参数：grads和variable，完成梯度更新
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        if step % 100 == 0:
            print(epoch,step,'losses:',float(loss))
            
    
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
    total_num=0
    total_correct=0
    for x,y in test_db:
        logits=model(x)
        y_onehot = tf.one_hot(y,depth = 100)
        loss=tf.losses.categorical_crossentropy(y_onehot,logits)
        loss=tf.reduce_mean(loss)
        
        #prob=tf.nn.softmax(logits,axis=1)
        pred=tf.argmax(logits,axis=1)
        pred=tf.cast(pred,dtype=tf.int32)
        correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct=tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)
        
        test_accuracy(correct/x.shape[0])
        test_loss(loss)
    
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))

    
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

model.save_weights('resattention.h5')

