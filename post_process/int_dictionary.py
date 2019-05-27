import argparse
import codecs
import json
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_input', default="./config/ner0.json")
    parser.add_argument('--raw_dev', default='./config/ner0.json')
    parser.add_argument('--raw_test', default='./config/ner0.json')
    parser.add_argument('--csv_output', default="./config/ner0.json")
    parser.add_argument('--threshold', type=int, default=2)
    args = parser.parse_args()

    with open(args.dict_input, 'r') as fin:
        entity_list = json.load(fin)

    new_entity_list = list()
    entity_dict = dict()
    for name, tag in entity_list:
        tmp_name = ' '.join(name)
        if tmp_name not in entity_dict:
            new_entity_list.append([name, tag])
            entity_dict[tmp_name] = {tag: 1}
        else:
            entity_dict[tmp_name][tag] = entity_dict[tmp_name].get(tag, 0) + 1

    small_dict = dict()
    for k, v in entity_dict.items():
        if len(v) == 1 and next(iter(v.items()))[1] > args.threshold:
            small_dict[k] = v

    entity_list = [tup for tup in new_entity_list if ' '.join(tup[0]) in small_dict]

    entity_list.sort(key=lambda t: -len(t[0]))

    with codecs.open(args.csv_output, 'w', 'utf-8') as fout:

        fout.write('Filename,Start,End,Type,Score,Surface\n')

        with codecs.open(args.raw_dev, 'r', 'utf-8') as fin:

            line_index = 1
            word_list = list()

            for line in fin:
                if line and not line.isspace():
                    cl = line.split()[0]
                    word_list.append([cl, line_index])
                else:
                    pred = ["O"] * len(word_list)
                    str_lst = [tup[0] for tup in word_list]

                    for name, tag in entity_list:
                        name_lst = name
                        starts = [i for i, j in enumerate(str_lst[:- len(name_lst) - 1]) if j == name_lst[0] and name_lst == str_lst[i: i+len(name_lst)]]
                        for s in starts:
                            if pred[s] == 'O' and len(set(pred[s:s+len(name_lst)])) == 1:
                                pred[s] == "B-" + tag
                                for i in range(s+1, s+len(name_lst)):
                                    pred[i] = "I-" + tag
                                phrase_name = '"' + ' '.join(str_lst[s:s + len(name_lst)]).replace(',', '<comma>').replace('"', '<qu>') + '"'
                                fout.write(','.join(["dev.txt", str(word_list[s][1]), str(word_list[s+len(name_lst)-1][1]), tag, "1.0", phrase_name])+'\n')
                    word_list = list()

                line_index += 1

        with codecs.open(args.raw_test, 'r', 'utf-8') as fin:

            line_index = 1
            word_list = list()

            for line in fin:
                if line and not line.isspace():
                    cl = line.split()[0]
                    word_list.append([cl, line_index])
                else:
                    pred = ["O"] * len(word_list)
                    str_lst = [tup[0] for tup in word_list]

                    for name, tag in entity_list:
                        name_lst = name
                        starts = [i for i, j in enumerate(str_lst[:- len(name_lst) - 1]) if j == name_lst[0] and name_lst == str_lst[i: i+len(name_lst)]]
                        for s in starts:
                            if pred[s] == 'O' and len(set(pred[s:s+len(name_lst)])) == 1:
                                pred[s] == "B-" + tag
                                for i in range(s+1, s+len(name_lst)):
                                    pred[i] = "I-" + tag
                                phrase_name = '"' + ' '.join(str_lst[s:s + len(name_lst)]).replace(',', '<comma>').replace('"', '<qu>') + '"'
                                fout.write(','.join(["test.txt", str(word_list[s][1]), str(word_list[s+len(name_lst)-1][1]), tag, "1.0", phrase_name])+'\n')
                    word_list = list()

                line_index += 1
