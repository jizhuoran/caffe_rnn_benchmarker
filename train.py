# -*- coding: utf-8 -*-

import caffe
import numpy as np
import random
import collections


caffe.set_device(1)
caffe.set_mode_gpu()
solver_poem = caffe.AdamSolver('solver_poem.prototxt')


def process_poems(file_name):

    poems = []
    frequence_list = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')

                content = content.replace(' ', '')

                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if not len(content)== 48:
                    continue

                content =  content.replace('。', '，')
                content = content[:-1] + '。'

                poems.append(content)

            except ValueError as e:
                pass

    poems = poems[:10000]

    count_pairs = sorted(collections.Counter([char for poem in poems for char in poem]).items(), key=lambda x: -x[1])
    word_int_map = {char:index for index, (char, count) in enumerate(count_pairs)}

    poems_vector = [list(map(lambda x: word_int_map[x], poem)) for poem in poems]

    random.shuffle(poems_vector)

    return poems_vector, word_int_map


def generate_batch(batch_size, poems_vec, word_to_int):
    
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        x_data = np.array(poems_vec[start_index:end_index])
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]

        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def main():
    
    poems_vector, word_to_int = process_poems('poems.txt')


    int_to_word = dict ( (v,k) for k, v in word_to_int.items() )

    data_clip = np.ones((48, 64))
    data_clip[0, :] = np.zeros((64))


    
    for epoch in range(0, 100000):
        random.shuffle(poems_vector)
        batches_inputs, batches_outputs = generate_batch(64, poems_vector, word_to_int)
        batch = 0
        for batches_input, batches_output in zip(batches_inputs, batches_outputs):
            
            solver_poem.net.blobs['data'].data[...] = np.array(batches_input).reshape(64, 48).transpose().reshape(48,64,1)
            solver_poem.net.blobs['label'].data[...] = np.array(batches_output).reshape(64, 48).transpose().reshape(3072,1)
            solver_poem.net.blobs['clip'].data[...] = data_clip

            solver_poem.net.forward()
            loss = solver_poem.net.blobs['loss'].data[...]
            solver_poem.net.clear_param_diffs()
            solver_poem.net.backward()
            solver_poem.apply_update()

            batch += 1

            if batch % 50 == 0:
                outputs = np.argmax(solver_poem.net.blobs['output_word'].data[...], axis = 1)
                acc = np.sum(outputs == np.array(batches_output).reshape(64, 48).transpose().reshape(3072)) / 3072
                
                print('Epoch: {}, batch: {}, training loss: {}, the acc is {}'.format(epoch, batch, loss, acc))
        


if __name__ == '__main__':
    main()
