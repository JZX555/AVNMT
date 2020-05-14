# encoding=utf8
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from hyper_and_conf import hyper_fn


def percent(current, total):

    if current == total - 1:
        current = total
    bar_length = 20
    hashes = '#' * int(current / total * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [%s] %d%%" %
                     (hashes + spaces, int(100 * current / total)))


def visualize_position_encoding(data):
    plt.pcolormesh(data, cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


# def plot_attention_weights_with_tokens(attention, sentence, result, layer):
#     fig = plt.figure(figsize=(16, 8))
#
#     sentence = tokenizer_pt.encode(sentence)
#
#     attention = tf.squeeze(attention[layer], axis=0)
#
#     for head in range(attention.shape[0]):
#         ax = fig.add_subplot(2, 4, head + 1)
#
#         # plot the attention weights
#         ax.matshow(attention[head][:-1, :], cmap='viridis')
#
#         fontdict = {'fontsize': 10}
#
#         ax.set_xticks(range(len(sentence) + 2))
#         ax.set_yticks(range(len(result)))
#
#         ax.set_ylim(len(result) - 1.5, -0.5)
#
#         ax.set_xticklabels(['<start>'] +
#                            [tokenizer_pt.decode([i])
#                             for i in sentence] + ['<end>'],
#                            fontdict=fontdict,
#                            rotation=90)
#
#         ax.set_yticklabels([
#             tokenizer_en.decode([i])
#             for i in result if i < tokenizer_en.vocab_size
#         ],
#                            fontdict=fontdict)
#
#         ax.set_xlabel('Head {}'.format(head + 1))
#
#     plt.tight_layout()
#     plt.show()
def plot_attention_weights_with_tokens(attention, sentence, result, layer=''):
    fig = plt.figure(figsize=(16, 8))
    # sentence = tokenizer_pt.encode(sentence)
    #
    # attention = tf.squeeze(attention[layer][0], axis=0)
    attention = attention[layer][0]
    sentence = ['<start>'] + [str(i) for i in sentence] + ['<end>']
    result = [str(i) for i in result]
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 5}

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(sentence, fontdict=fontdict, rotation=90)

        ax.set_yticklabels(result, fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def plot_learning_rate():
    total_step = 200000
    t = np.arange(0, total_step, 1)
    s = []
    for i in range(0, total_step):
        s.append(hyper_fn.get_learning_rate(
            2, 1024, step=i, learning_rate_warmup_steps=3000))

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='steps', ylabel='learining rate',
           title='learing rate')
    ax.grid()

    fig.savefig("lr.png")
    plt.show()


plot_learning_rate()
