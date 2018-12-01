import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from main1 import model

max_size = len(model.wv.vocab)
w2v = np.zeros((max_size,model.layer1_size))
print(w2v)

with open("metadata.tsv", 'w+') as file_metadata:
    for i,word in enumerate(model.wv.index2word[:max_size]):
        w2v[i] = model.wv[word]
        file_metadata.write(word + '\n')

print(w2v)


sess = tf.InteractiveSession()
#Let us create a 2D tensor called embedding that holds our embeddings.
with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable=False, name='embedding')

tf.global_variables_initializer().run()
log_dir = 'log-1'

# let us create an object to Saver class which is actually used to 
#save and restore variables to and from our checkpoints
saver = tf.train.Saver()
# using file writer, we can save our summaries and events to our event file.
writer = tf.summary.FileWriter(log_dir, sess.graph)

# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embedding'
embed.metadata_path = "metadata.tsv"

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)

print(saver.save(sess, log_dir+'/model.ckpt', global_step=max_size))



