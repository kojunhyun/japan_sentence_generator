# -*- coding: utf-8 -*-

import random
import collections
import japan_corpus_preprocessing as ppc
import numpy as np

# random seed는 동일 실험조건을 충족시키기 위해 고정시켜 줍니다.
random.seed(777)

def _read_ch_file(filename):
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(tr_path, voca_path, voca_size=None):

    tr_data = _read_ch_file(tr_path)
    counter = collections.Counter(tr_data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, word_count = zip(*count_pairs)
    words = list(words)

    if not voca_size is None:
        words = words[:voca_size-1]

    words.append('<unk>')
    
    with open(voca_path + 'vocab.txt', 'w', encoding='utf-8') as w_f:

        for i in range(0, len(words)):
            w_f.write(words[i] + '\t')
            w_f.write(str(i) + '\n')
                
    
    word_to_id = dict(zip(words, range(len(words))))
    print('vocabulray size : ', len(word_to_id))

    return word_to_id


def _id_mapping(filename, ch_to_id):
    
    data = _read_ch_file(filename)

    # vocab에 없는 문자는 <unk>의 id를 사용한다.
    id = []
    for ch in data:
        if ch in ch_to_id:
            id.append(ch_to_id[ch])
        else:
            id.append(ch_to_id['<unk>'])

    return id


def _tr_va_split(data_path):
    extension_index = data_path.find('.')  
    #print(data_path[:extension_index])

    with open(data_path, 'r', encoding='utf-8') as f:        
        lines = f.readlines()

    refined_data = ppc._japan_data_refine(lines)


    total_line = len(refined_data)
    train_line = int(total_line *0.7)
    
    with open(data_path[:extension_index]+'_train.txt', 'w', encoding='utf-8') as f:
        for i in range(0, train_line):
            for char in lines[i]:
                f.write(char + ' ')
            
      
    with open(data_path[:extension_index]+'_valid.txt', 'w', encoding='utf-8') as f:
        for i in range(train_line, total_line):
            for char in lines[i]:
                f.write(char + ' ')
            
    

    tr_path = data_path[:extension_index] + '_train.txt'
    va_path = data_path[:extension_index]+'_valid.txt'

    return tr_path, va_path 

def corpus_raw_data(set_voca_size, voca_mode, data_path=None, vocab_path=None):
    '''
    
    Load corpus raw data from data path "data_path".

    Reads corpus data files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    
    Args:
    set_voca_size: setting vocabulary size
    voca_mode: total vocabulary size or setting vocabulary size select - True, False
    data_path: string path to the file 
    vocab_path: string path to the save file

    Returns:
    tuple (train_data, valid_data, vocabulary, word_to_id)
    where each of the data objects can be passed to CorpusIterator.
    '''
    
    
    #tmp_tr_data, tmp_va_data = _tr_va_split(data_path)
    tr_path, va_path = _tr_va_split(data_path)
    
    if voca_mode:
        char_to_id = _build_vocab(tr_path, vocab_path)
    else:
        char_to_id = _build_vocab(tr_path, vocab_path, set_voca_size)
    train_data = _id_mapping(tr_path, char_to_id)
    valid_data = _id_mapping(va_path, char_to_id)
    #char_to_id = _build_vocab(tmp_tr_data, vocab_path)
    #train_data = _id_mapping(tmp_tr_data, char_to_id)
    #valid_data = _id_mapping(tmp_va_data, char_to_id)

    print('train len : ', len(train_data))
    print('valid len : ', len(valid_data))

    vocabulary_size = len(char_to_id)

    if voca_mode is False:
        if set_voca_size < vocabulary_size:
            #print(char_to_id[0])
            #char_to_id = char_to_id[:set_voca_size]
            vocabulary_size = set_voca_size
        #else:
        #    vocabulary_size = len(char_to_id)
 
    
    return np.array(train_data), np.array(valid_data), vocabulary_size, char_to_id


def corpus_iterator(data, batch_size, num_steps):
    
    '''Iterate on the tensorflow input data.
    
    This generates batch_size pointers into the char-corpus data, and allows
    minibatch iteration along these pointers.

    Args:
    data: outputs from corpus_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

    Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

    Raises:
    ValueError: if batch_size or num_steps are too high.
    
    '''

    # mini-batch로 기울기를 조절할 때에 데이터 수가 적으면 기울기가 크게 변할 수 있는 가능성이 있어
    # 마지막 batch*num_steps 크기 만큼 데이터가 없으면 버린다.

    np_data = np.array(data, dtype=np.int32)

    data_len = len(np_data)
    print('data len : ')
    print(data_len)

    batch_len = int(data_len / batch_size)
    print('batch_len : ')
    print(batch_len)

    batch_data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        batch_data[i] = np_data[batch_len * i:batch_len * (i + 1)]
    
    epoch_size = int((batch_len - 1) / num_steps)

    #epoch_size = int(data_len / (batch_size * num_steps) - 1)
    print('epoch size : ')
    print(epoch_size)


    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    
    for i in range(epoch_size):
        x = batch_data[:, i*num_steps:(i+1)*num_steps]
        y = batch_data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)
    """

    for i in range(epoch_size):
        data_x = np_data[i*(batch_size * num_steps):i*(batch_size * num_steps)+(batch_size * num_steps)]
        data_y = np_data[i*(batch_size * num_steps)+1:i*(batch_size * num_steps)+(batch_size * num_steps)+1]

        x = np.reshape(data_x, [batch_size, num_steps])
        y = np.reshape(data_y, [batch_size, num_steps])

        yield (x, y)
    """

def main():
    data_path = 'data\sns_text2.txt'
    vocab_path = 'model\\'
    tr_path, va_path = _tr_va_split(data_path)

    word_to_id = _build_vocab(tr_path, vocab_path)
    train_data = _id_mapping(tr_path, word_to_id)
    print(train_data[:10])
    print(train_data[-10:])
    print(len(train_data))


if __name__ == "__main__":
    main()
    
