# -*- coding: utf-8 -*-
import os
import sys
import argparse
import unicodedata


def duplex_char_remove(text):
    before_letter = ''
    total_text = ''
    before_count = 0
    for current_letter in text:
        if current_letter == before_letter:
            before_count += 1            
            if before_count > 1:
                pass
            else:
                total_text += current_letter
        else:
            total_text += current_letter
            before_letter = current_letter
            before_count = 0
    return total_text


def japan_paragraph_to_sentence(paragraph):
    # 일본어 문단을 문장으로 split
    # japan paragraph to sentence
    #print(text)
    setting_paragraph = []
    sum_letter = ''
    for letter in paragraph:
        sum_letter += letter
        if letter == u'。' or letter == '!' or letter == '?':
            setting_paragraph.append(sum_letter.strip())
            sum_letter = ''

    return setting_paragraph  


def japan_data_refine(lines_input):
    
    refined_data = []

    for line in lines_input:
        # NFKC 정규화        
        line = unicodedata.normalize('NFKC', line)

        # 문단 -> 문장으로 나눠서 정제하기 위한 setting
        line = line.replace(u'。', u'。 ')
        line = line.replace('!', '! ')
        line = line.replace('?', '? ')
        setting_line = line.split()

        # 정제 후 다시 문단으로 합치기
        re_paragraph = ''

        for split_line in setting_line:
            refine_rule = [0,0,0,0,0,0,0]
            sp_line = split_line.lower()

            # 1라인에 문자 하나있는 것들은 문장이 아니기 때문에 제외(sequence data가 필요)
            if len(sp_line) == 1:
                pass
            else:
                # url 주소 등 개인 domain 부분들 제거하기 위한 rule
                if sp_line.find('http') == -1:
                    refine_rule[0] = 1
                if sp_line.find('https') == -1:
                    refine_rule[1] = 1
                if sp_line.find('www.') == -1:
                    refine_rule[2] = 1
                if sp_line.find('pic.twitter') == -1:
                    refine_rule[3] = 1
                if sp_line.find('#') == -1:
                    refine_rule[4] = 1
                if sp_line.find('@') == -1:
                    refine_rule[5] = 1
                if sp_line.find('.jp') == -1:
                    refine_rule[6] = 1
                
                if 0 not in refine_rule:
                    re_paragraph += split_line + ' '

        if not re_paragraph == '':
            # 중복 문자 2개로 고정
            remove_data = duplex_char_remove(re_paragraph)
            sentence_list = japan_paragraph_to_sentence(remove_data)

            for sentence in sentence_list:
                refined_data.append(sentence)

    return refined_data


def main(input_path, output_path):
    
    with open(input_path, 'r', encoding='utf-8') as r_f:
        lines_input_data = r_f.readlines()

    refined_data = japan_data_refine(lines_input_data)

    with open(output_path, 'w', encoding='utf-8') as w_f:
        for rd in refined_data:
            w_f.write(rd + '\n')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', help='japan corpus insert')
    parser.add_argument('--output', nargs='+', help='output corpus name')
    args = parser.parse_args()


    if args.input is None:
        print('input data is not exist')
        sys.exit()
    elif args.output is None:
        print('output data is not defined')
        sys.exit()
    elif len(args.input) is not len(args.output):
        print('number of files is not matching')
        sys.exit()
    else:
        # file check
        for i in range(len(args.input)):
            d_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.input[i])
            if os.path.exists(d_path) is True:
                main(args.input[i], args.output[i])
            else:
                print('%s data is not exist' %d_path)
