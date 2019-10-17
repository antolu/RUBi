import tensorflow as tf
#from baseline_ques_enc import encode_ques_data
import numpy as np


#### q_emb = encode_ques_data(q)
# q, _ = txt_enc.rnn(q_emb) ?
# shape q = 2400 x no_questions
#### q = q_emb


#dummy question embeddings
q = np.array([100, 100])

# add attention module to the text_encoder
attention_module = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='linear'),
    tf.keras.layers.ReLU(max_value=1.0, threshold=-1.0),
    tf.keras.layers.Dense(2, activation='linear'),
    tf.keras.layers.Softmax(),  # this was mask_softmax(q_att, l)

])

attention_module.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
attention_module.fit(q)

print(attention_module.summary())

q_att = attention_module(q)

# something with if q_att.size > 2

q_att = q_att.expand_as(q)
q = q_att * q
q = q.sum(1)
