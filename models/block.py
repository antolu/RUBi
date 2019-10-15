from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)  # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list


def get_chunks(x, sizes):
    out = []
    begin = 0
    print("sizes: ", sizes)
    tf.print(x)
    # print("x: ", x.eval())
    print("y: ", x[:,0:sizes[0]])
    for s in sizes:
        # y = x.narrow(1, begin, s)
        y = x[:, begin:begin+s]
        out.append(y)
        begin += s
    return out


class Block(Model):
    def __init__(self,
                 input_dims,
                 output_dim,
                 mm_dim=1600,
                 chunks=20,
                 rank=15,
                 shared=False,
                 dropout_input=0.,
                 dropout_pre_lin=0.,
                 dropout_output=0.,
                 pos_norm='before_cat'):
        super(Block, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert (pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm

        # Modules
        self.linear0 = Dense(input_shape=[input_dims[0]], units=mm_dim, activation='linear')
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = Dense(input_shape=[input_dims[1]], units=mm_dim, activation='linear')
        self.merge_linears0, self.merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = Dense(input_shape=[size], units=size * rank, activation='linear')
            self.merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = Dense(input_shape=[size], units=size * rank, activation='linear')
            self.merge_linears1.append(ml1)

        self.linear_out = Dense(input_shape=[mm_dim], units=output_dim, activation='linear')
        # self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def call(self, x):
        # x0 = self.linear0(x[0])
        # x1 = self.linear1(x[1])
        
        print("x: ", x)
        print("-------------------------------------")
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        print("x0: ", x0)
        print("-------------------------------------")
        print("x1: ", x1)
        # bsize = x1.size(0)
        bsize = x1.shape[0]
        if self.dropout_input > 0:
            x0 = tf.nn.dropout(x0, rate=self.dropout_input)
            x1 = tf.nn.dropout(x1, rate=self.dropout_input)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)),
                                    self.merge_linears0,
                                    self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c) # bsize x split_size*rank
            # m = m.view(bsize, self.rank, -1)
            print("----------------------------")
            print("m: ", m)
            print(type(m))
            m = tf.reshape(m, [bsize, self.rank, -1])
            z = tf.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = tf.sqrt(tf.nn.relu(z)) - tf.sqrt(tf.nn.relu(-z))
                z = tf.math.l2_normalize(z)
            zs.append(z)
        z = tf.concat(zs, axis=1)
        if self.pos_norm == 'after_cat':
            z = tf.sqrt(tf.nn.relu(z)) - tf.sqrt(tf.nn.relu(-z))
            z = tf.math.l2_normalize(z)

        if self.dropout_pre_lin > 0:
            z = tf.nn.dropout(z, rate=self.dropout_pre_lin)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = tf.nn.dropout(z, rate=self.dropout_output)
        return z
