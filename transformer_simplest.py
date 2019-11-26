""" Transformer implementation in Tensorflow 2.
This is really just the decoder block stripped from the official GPT-2 implementation.
"""
# TODO kill the estimator class and replace with Keras
# TODO new dataset

import os
import numpy as np

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# os.environ['PATH'] = '$PATH:/usr/local/cuda-10.0/bin'
os.environ['CUDADIR']='/usr/local/cuda-10.0'
# os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64'

# export PATH=$PATH:/usr/local/cuda-10.0/bin
# export CUDADIR=/usr/local/cuda-10.0
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64

import tensorflow as tf
import tensorflow_datasets as tfds

#tf.config.experimental_run_functions_eagerly(True)

NUM_LABELS = 2

class Params:  # TODO move this to not be... here
    def __init__(self):
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.vocab_size = 8185
        self.num_heads = 16
        self.max_context = 1024
        self.embedding_dim = 512
        self.ffn_expansion = 4
        self.is_training = True
        self.num_blocks = 12


params = Params()


def trim_or_pad(example):
    example_length = tf.shape(input=example["text"])[0]
    example["text"] = tf.cond(pred=example_length < params.max_context,
                              true_fn=lambda: tf.pad(tensor=example["text"],
                                                     paddings=[[0, params.max_context - example_length]]),
                              false_fn=lambda: example["text"][:params.max_context])
    example["text"].set_shape([params.max_context])
    return {
        "text": example["text"],
        "label": example["label"]
    }



def dict_to_tuple(example):
    return (tf.cast(example['text'][0:params.max_context], tf.int32), tf.cast(example['label'], tf.float32))


# Create tf.Data object for feeding input data
mode = 'train'
dataset, info = tfds.load(name='imdb_reviews/subwords8k', with_info=True, split='train')

# TODO this is better but strip it down
# input_data = dataset.take(50000) \
#             .shuffle(1000) \
#             .map(trim_or_pad, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
#             .repeat() \
#             .batch(params.batch_size) \
#             .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#input_data = dataset.take(50000).shuffle(1000).map(trim_or_pad).repeat(1).batch(params.batch_size)
input_data = dataset.take(50000).shuffle(1000).map(dict_to_tuple)\
    .padded_batch(batch_size=params.batch_size,padded_shapes=([None],[]), padding_values=(0, 0.0))\
    .repeat(1)


def get_embeddings(input_data, hparams):
    for features in input_data:
        input_sequence = features[0]  # [[ ... ] ... ]
        label = features[1]

    embedded_inputs = norm(embed_tokens(input_sequence, hparams))

    return embedded_inputs, label
#
#ugh, label = get_embeddings(input_data, params)


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(input=x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value=value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, one_hot=False):
    nsteps = tf.shape(input=tokens)[1]
    steps = tf.range(nsteps)

    if one_hot:
        return tf.one_hot(steps, nsteps)

    return steps


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model, name='PositionalEncodingLayer'):
        super(PositionalEncodingLayer, self).__init__()
        # Create a table of all possible positional encodings the model will see
        self.pos_encoding = self.compute_positional_encoding(max_position, d_model, name=name)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def compute_positional_encoding(self, position, d_model, name):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32, name=name)

    def call(self, seq_len):
        # Select relevant positions up to the sequence length
        return self.pos_encoding[:, :seq_len, :]


def embed_tokens(tokens, hparams, *, scope="embeddings"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        print(tokens.shape)
        position_embeddings = tf.compat.v1.get_variable(
            'position_embeddings', [hparams.max_context, hparams.embedding_dim],
            initializer=tf.compat.v1.random_normal_initializer(stddev=0.01))
        token_embeddings = tf.compat.v1.get_variable(
            'token_embeddings', [hparams.vocab_size, hparams.embedding_dim],
            initializer=tf.compat.v1.random_normal_initializer(stddev=0.02))

    pe = tf.einsum("sc,cd->sd", positions_for(tokens, one_hot=True),
                   position_embeddings)
    pe = expand_tile(pe, tf.shape(input=tokens)[0])
    return tf.gather(token_embeddings, tokens) + pe


def norm(x, *, scope=None, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.compat.v1.variable_scope(scope, default_name="norm"):
        n_state = x.shape[-1]  # .value
        g = tf.compat.v1.get_variable('g', [n_state], initializer=tf.compat.v1.constant_initializer(1))
        b = tf.compat.v1.get_variable('b', [n_state], initializer=tf.compat.v1.constant_initializer(0))
        u = tf.reduce_mean(input_tensor=x, axis=axis, keepdims=True)
        s = tf.reduce_mean(input_tensor=tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + epsilon)
        x = x * g + b
        return x


def conv1d(x, scope, neurons, *, w_init_stdev=0.02):
    with tf.compat.v1.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.compat.v1.get_variable(
            'w', [1, nx, neurons],  # nx from shape_list, neurons = num_features
            initializer=tf.compat.v1.random_normal_initializer(stddev=w_init_stdev))
        b = tf.compat.v1.get_variable('b', [neurons], initializer=tf.compat.v1.constant_initializer(0))
        c = tf.reshape(
            tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, neurons])) + b,
            start + [neurons])
        return c


def mlp(x, scope, n_state, *, hparams):
    with tf.compat.v1.variable_scope(scope):
        assert int(n_state) - n_state == 0.0
        n_state = int(n_state)
        nx = x.shape[-1]  # .value
        #h = tf.nn.relu(conv1d(x, 'c_fc', n_state))
        h = tf.nn.relu(tf.keras.layers.Dense(n_state)(x))
        #h2 = conv1d(h, 'c_proj', nx)
        h2 = tf.keras.layers.Dense(nx)(h)
        return h2


class decoder_block(tf.keras.layers.Layer):
    # TODO: attention  dropout

    def __init__(self, num_heads, scope='decoder'):
        super(decoder_block, self).__init__()

        self.projections = [tf.keras.layers.Dense(params.embedding_dim * 3 // params.num_heads) for _ in range(num_heads)]
        #self.dense1 = tf.keras.layers.Dense(params.embedding_dim * 3 // params.num_heads)
        self.multihead_attn = tf.keras.layers.Dense(params.embedding_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.scope = scope

    def call(self, x):

        residual = x
        num_heads = params.num_heads

        all_heads = []
        for i in range(num_heads):
            #print("Shape of broken stuff: {}".format(x.shape))
            d = self.projections[i](x)
            #d = conv1d(x, scope=scope, neurons=params.embedding_dim * 3 // num_heads)  # (?, 64, 17)
            q, k, v = tf.split(d, num_or_size_splits=3, axis=-1)
            dim_k = tf.shape(k)[1]
            qk_t = tf.matmul(q, k, transpose_b=True) * tf.math.rsqrt(tf.cast(dim_k, dtype=tf.float32))
            x = tf.matmul(tf.nn.softmax(qk_t), v)
            all_heads.append(x)

        all_heads_tensor = tf.concat(all_heads, axis=-1)
        #multihead_attn = conv1d(all_heads_tensor, scope=scope + 'proj', neurons=params.embedding_dim)
        multihead_attn = self.multihead_attn(all_heads_tensor)

        x = multihead_attn
        x = x + residual
        #x = norm(x, scope=scope + 'norm')
        x = self.layernorm1(x)
        residual2 = x
        x = mlp(x, scope=self.scope + 'mlp', n_state=params.embedding_dim * params.ffn_expansion, hparams=params)
        x = x + residual2
        #x = norm(x, scope=scope + 'norm2')
        x = self.layernorm2(x)

        return x





# same as the below, but implemented with the imperative API
class TinyTransformer(tf.keras.Model):
    def __init__(self, params):
        super(TinyTransformer, self).__init__(self)
        self.hparams = params
        self.embedding_layer = tf.keras.layers.Embedding(params.vocab_size,
                                                    params.embedding_dim)
        self.positional_embeddings = PositionalEncodingLayer(max_position=params.max_context, d_model=params.embedding_dim)
        self.optimizer = tf.keras.optimizers.Adam()

        self.blocks = []
        for i in range(params.num_blocks):  # 4x more decoder blocks
            self.blocks.append(decoder_block(num_heads=params.num_heads, scope=str(i)))

        self.final_dense = tf.keras.layers.Dense(units=1)


    def call(self, inputs):
        #[batch_size, tokens]
        # Embed inputs
        embedded_inputs = self.embedding_layer(inputs)
        embedded_inputs_with_positions = embedded_inputs + self.positional_embeddings(tf.shape(inputs)[1])

        # From embeddings
        #x = decoder_block(embedded_inputs_with_positions, self.hparams, scope='b1')
        x = embedded_inputs_with_positions

        for block in self.blocks:
            x = block(x)

        #x = tf.reshape(x, [self.hparams.batch_size, tf.reduce_prod(input_tensor=x.shape[1:])])
        #x = x[:, 0]
        x = tf.reduce_mean(x, axis=1)  # TODO reduce better

        logits = self.final_dense(x)
        outputs = tf.nn.sigmoid(logits)

        return logits, outputs

    def compute_loss(self, logits, labels):
        #print(logits)
        #print(labels)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits[:, 0]))
        return loss


    @tf.function(input_signature=[tf.TensorSpec(shape=[params.batch_size, None], dtype=tf.int32), tf.TensorSpec(shape=[params.batch_size, ], dtype=tf.float32)])
    def train_step(self, inputs, targets):

        with tf.GradientTape() as tape:
            logits, outputs = self.call(inputs=inputs)
            loss = self.compute_loss(logits, targets)

            tf.print("Loss: {}".format(loss))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


    def train(self, input_data):
        for data in input_data:
            self.train_step(data[0], data[1])


model = TinyTransformer(params)
print("Compiling model")
#model.compile(optimizer='adam')
print("Successfully compiled model")
print("Getting embeddings")
#ugh, label = get_embeddings(input_data, params)
print("Training")
print('hello')


model.train(input_data)

# TODO replace this with the imperative API
'''
def model(features, hparams):
    input_sequence = features["text"]  # [[ ... ] ... ]
    label = features["label"]  # 0 or 1

    # Embedding stuff
    embedded_inputs = embed_tokens(input_sequence,
                                   hparams)
    embedded_inputs = norm(embedded_inputs)

    # Transformer stuff
    output_layer = decoder_block(embedded_inputs, hparams, scope='b1')
    output_layer = decoder_block(output_layer, hparams, scope='b2')
    output_layer = decoder_block(output_layer, hparams, scope='b3')
    output_layer = decoder_block(output_layer, hparams, scope='b4')
    output_layer = decoder_block(output_layer, hparams, scope='b5')
    output_layer = tf.reshape(output_layer, [hparams.batch_size, tf.reduce_prod(input_tensor=output_layer.shape[1:])])

    with tf.compat.v1.variable_scope("top_ffn", initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)):
        output_weights = tf.compat.v1.get_variable(
            "output_weights", [NUM_LABELS, output_layer.shape[1]])

        output_bias = tf.compat.v1.get_variable(
            "output_bias", [NUM_LABELS], initializer=tf.compat.v1.zeros_initializer())

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(label, depth=NUM_LABELS, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(input=log_probs, axis=-1, output_type=tf.int32), name="output")
    per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(input_tensor=per_example_loss)

    return predicted_labels, loss
'''









'''
def model_fn(features, labels, mode, params):
    params.is_training = mode == tf.estimator.ModeKeys.TRAIN
    predicted_labels, loss = model(features, params)
    label_ids = features["label"]

    # Calculate evaluation metrics.
    def metric_fn(label_ids, predicted_labels):
        accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
        # f1_score = tf.contrib.metrics.f1_score(
        #     label_ids,
        #     predicted_labels)
        auc = tf.compat.v1.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.compat.v1.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.compat.v1.metrics.precision(
            label_ids,
            predicted_labels)
        true_pos = tf.compat.v1.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.compat.v1.metrics.true_negatives(
            label_ids,
            predicted_labels)
        false_pos = tf.compat.v1.metrics.false_positives(
            label_ids,
            predicted_labels)
        false_neg = tf.compat.v1.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            # "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

    eval_metrics = metric_fn(label_ids, predicted_labels)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        #gs = tf.compat.v1.train.get_global_step()
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(
            loss, global_step=tf.compat.v1.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)


# params = tf.contrib.training.HParams(
#       learning_rate=1e-4,
#       batch_size=32,
#       vocab_size=8185,
#       num_heads=1,
#       max_context=64,
#       embedding_dim=16,
#       ffn_expansion=4,
#       is_training=True)

# HN replace HParams with a params class


cfg = tf.estimator.RunConfig(save_checkpoints_steps=1000)

if tf.io.gfile.exists('results/model'):
    tf.io.gfile.rmtree('results/model')

estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn("train", params), max_steps=2000)
eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn("test", params), throttle_secs=0)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

print()
'''