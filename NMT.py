"""
This is based on Assignment 3 Open Domain Dialogue System
https://docs.google.com/document/d/1GJfn2B6EI8JueDiBwzTAdD34d6pC99BSt6vldOmUCPQ/edit#heading=h.mg4k7iiiszlp
"""
import sys
import re
import numpy as np
import random
import tensorflow as tf
import os
import nltk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

THRESHOLD = 1
ENC_VOCAB = 41303
DEC_VOCAB = 18778
NUM_SAMPLES = 18777
NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 256
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3
LR = 0.5
MAX_GRAD_NORM = 5.0
BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]
MAX_ITERATION = 30001
PROCESSED_PATH='Data'
deviceType = "/gpu:0"

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words

def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(PROCESSED_PATH, filename)
    out_path = os.path.join(PROCESSED_PATH, 'vocab.{}'.format(filename[-2:]))

    vocab = {}
    with open(in_path, 'r') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < THRESHOLD:
                break
            f.write(word + '\n')
            index += 1

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(PROCESSED_PATH, out_path), 'w')

    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'vi':  # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'vi':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.en')
    build_vocab('train.vi')
    token2id('train', 'en')
    token2id('train', 'vi')
    token2id('tst2012', 'en')
    token2id('tst2012', 'vi')

def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(os.path.join(PROCESSED_PATH, dec_filename), 'r')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in BUCKETS]
    i = 0
    while encode and decode:
        # if (i + 1) % 10000 == 0:
        #     print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets

def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us
    choose a random bucket later on.
    """
    print("Bucket:",BUCKETS)
    test_buckets = load_data('tst2012_ids.en', 'tst2012_ids.vi')
    data_buckets = load_data('train_ids.en', 'train_ids.vi')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(BUCKETS))]
    print("Number of samples in each bucket:", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale


def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

def _pad_input(input_, size):
    return input_ + [PAD_ID] * (size - len(input_))


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs

def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_masks), decoder_size))

def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(BUCKETS))
                if BUCKETS[b][0] >= length])

def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    # print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


class ChatBotModel (object):
    def __init__ (self, isTrain=1):
        with tf.device(deviceType):
            # print('Initialize new model')
            self.fw_only = isTrain
            if isTrain:
                self.batch_size=1
            else:
                self.batch_size=BATCH_SIZE

            # def _create_placeholders(self):
            self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                                   for i in range(BUCKETS[-1][0])]
            self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                                   for i in range(BUCKETS[-1][1] + 1)]
            self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                                  for i in range(BUCKETS[-1][1] + 1)]
            # Our targets are decoder inputs shifted by one (to ignore <GO> symbol)
            self.targets = self.decoder_inputs[1:]

            # def _inference(self):
            # If we use sampled softmax, we need an output projection.
            # Sampled softmax only makes sense if we sample less than vocabulary size.
            if NUM_SAMPLES > 0 and NUM_SAMPLES < DEC_VOCAB:
                w = tf.get_variable('proj_w', [HIDDEN_SIZE, DEC_VOCAB])
                b = tf.get_variable('proj_b', [DEC_VOCAB])
                self.output_projection = (w, b)
            def sampled_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(weights=tf.transpose(w),
                                                  biases=b,
                                                  inputs=logits,
                                                  labels=labels,
                                                  num_sampled=NUM_SAMPLES,
                                                  num_classes=DEC_VOCAB)

            self.softmax_loss_function = sampled_loss

            single_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
            self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(NUM_LAYERS)])

            # def _create_loss(self):
            def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
                setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=ENC_VOCAB,
                    num_decoder_symbols=DEC_VOCAB,
                    embedding_size=HIDDEN_SIZE,
                    output_projection=self.output_projection,
                    feed_previous=do_decode)

            if self.fw_only:
                self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    self.targets,
                    self.decoder_masks,
                    BUCKETS,
                    lambda x, y: _seq2seq_f(x, y, True),
                    softmax_loss_function=self.softmax_loss_function)
                # If we use output projection, we need to project outputs for decoding.
                if self.output_projection:
                    for bucket in range(len(BUCKETS)):
                        self.outputs[bucket] = [tf.matmul(output,
                                                          self.output_projection[0]) + self.output_projection[1]
                                                for output in self.outputs[bucket]]
            else:
                self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    self.targets,
                    self.decoder_masks,
                    BUCKETS,
                    lambda x, y: _seq2seq_f(x, y, False),
                    softmax_loss_function=self.softmax_loss_function)

            # def _creat_optimizer(self):
            with tf.variable_scope('training') as scope:
                self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

                if not self.fw_only:
                    self.optimizer = tf.train.GradientDescentOptimizer(LR)
                    trainables = tf.trainable_variables()
                    self.gradient_norms = []
                    self.train_ops = []
                    for bucket in range(len(BUCKETS)):
                        clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                                  trainables),
                                                                     MAX_GRAD_NORM)
                        self.gradient_norms.append(norm)
                        self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                             global_step=self.global_step))


    def train(self, data_buckets, train_buckets_scale):
        # """ Train the bot """
        print("ENC_VOCAB:",ENC_VOCAB)
        print("DEC_VOCAB:",DEC_VOCAB)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            print('Running session')
            iteration = 0
            total_loss=0
            while iteration < MAX_ITERATION:
                bucket_id = _get_random_bucket(train_buckets_scale)
                encoder_inputs, decoder_inputs, decoder_masks = get_batch(data_buckets[bucket_id],
                                                                               bucket_id,
                                                                               batch_size=BATCH_SIZE)
                _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
                total_loss += step_loss

                if iteration % 2000 ==0:
                    if iteration==0:
                        print('Iter {}: loss {}'.format(iteration, total_loss ))
                    else:
                        print('Iter {}: loss {}'.format(iteration, total_loss / 1000))
                    total_loss=0
                iteration += 2

            saver = tf.train.Saver()
            saver_path = saver.save(sess, "./model/model.ckpt")
            print("Model saved in file: ", saver_path)

        pass


    def chat(self, token_ids):
        """ in test mode, we don't to create the backward path
            """
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, "./model/model.ckpt")
            # print("Loading model...")
            # print("Load model success")

            # Which bucket does it belong to?
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = get_batch([(token_ids, [])],
                                                                            bucket_id,
                                                                            batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            return output_logits

    def test(self):
        _, enc_vocab = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.en'))
        inv_dec_vocab, _ = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.vi'))
        hypothesis_file = open(os.path.join(PROCESSED_PATH, 'tst2012.en'), 'r')
        reference_file = open(os.path.join(PROCESSED_PATH, 'tst2012.vi'), 'r')
        BLEUscore=[]
        max_length = BUCKETS[-1][0]
        count=0
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            for (r, h) in zip(reference_file,hypothesis_file):
                # re_tst= r.readline()
                re_tst = r.strip()
                # hy_tst = h.readline()
                hy_tst = h.strip()
                if len(hy_tst) > max_length:
                    continue
                if count < 200:
                    token_ids = sentence2id(enc_vocab,str(hy_tst))
                    output_logits = self.chat(token_ids)
                    response = _construct_response(output_logits,inv_dec_vocab)
                    BLEUscore.append(nltk.translate.bleu_score.sentence_bleu([re_tst.split()],response.split(),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1))
                    count+=1
                    if count < 20:
                        print (response)
            totalBLEU = np.sum(BLEUscore)/count
            print("Average BLEU：",totalBLEU)
        pass

    def translate(self):
        print("Loading model...")
        _, enc_vocab = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.en'))
        inv_dec_vocab, _ = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.vi'))
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            # Decode from standard input.
            max_length = BUCKETS[-1][0]
            while True:
                line = _get_user_input()
                if len(line) > 0 and line[-1] == '\n':
                    line = line[:-1]
                if line == '':
                    break
                token_ids = sentence2id(enc_vocab, str(line))
                if (len(token_ids) > max_length):
                    print('Max length I can handle is:', max_length)
                    continue
                output_logits=self.chat(token_ids)
                response = _construct_response(output_logits, inv_dec_vocab)
                print(response)
        pass

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


############################################
########Your main Function here#############
if __name__ == "__main__":
    if sys.argv[1] == "train":
        process_data()
        # """ Train the bot """
        test_buckets, data_buckets, train_buckets_scale = _get_buckets()
        # in train mode, we need to create the backward path, so forwrad_only is False
        model = ChatBotModel(False)
        model.train(data_buckets,train_buckets_scale)

    elif sys.argv[1] == 'test':
        #""" Test the bot"""
        model = ChatBotModel(True)
        model.test()

    elif sys.argv[1] == 'translate':
        model = ChatBotModel(True)
        model.translate()

    else:
        print('Command error')


