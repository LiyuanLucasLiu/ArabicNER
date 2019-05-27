import argparse
import codecs
import json
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_csv', default='./config/ner0.json')
    parser.add_argument('--dict_csv', default='./config/ner0.json')
    parser.add_argument('--csv_output', default="./config/ner0.json")
    args = parser.parse_args()

    with codecs.open(args.model_csv, 'r', 'utf-8') as fin:
        reader = csv.DictReader(fin)
        all_lines = [line for line in reader]
        model_dev_lines = [row for row in all_lines if row['Filename'] == 'dev.txt']
        model_test_lines = [row for row in all_lines if row['Filename'] == 'test.txt']

    model_dev_lines.sort(key = lambda t: int(t['Start']))
    model_test_lines.sort(key = lambda t: int(t['Start']))

    with codecs.open(args.dict_csv, 'r', 'utf-8') as fin:
        reader = csv.DictReader(fin)
        all_lines = [line for line in reader]
        dict_dev_lines = [row for row in all_lines if row['Filename'] == 'dev.txt']
        dict_test_lines = [row for row in all_lines if row['Filename'] == 'test.txt']
    
    dict_dev_lines.sort(key=lambda t: int(t['Start']))
    dict_test_lines.sort(key=lambda t: int(t['Start']))

    with codecs.open(args.csv_output, 'w', 'utf-8') as fout:

        fout.write('Filename,Start,End,Type,Score,Surface\n')

        for source, target in zip([dict_dev_lines, dict_test_lines], [model_dev_lines, model_test_lines]):
            pre_end = -1
            cursor = 0

            for line in source:
                while cursor < len(target) and int(target[cursor]['Start']) < int(line['Start']):
                    fout.write(','.join([target[cursor]['Filename'], target[cursor]['Start'], target[cursor]['End'], target[cursor]['Type'], target[cursor]['Score'], '"'+target[cursor]['Surface'].strip('"')+'"']) + '\n')
                    pre_end = int(target[cursor]['End'])
                    cursor += 1

                if pre_end < int(line['Start']) and int(line['End']) < int(target[cursor]['Start']):
                    fout.write(','.join([line['Filename'], line['Start'], line['End'], line['Type'], line['Score'], '"' + line['Surface'].strip('"') + '"']) + '\n')

            while cursor < len(target):
                    fout.write(','.join([target[cursor]['Filename'], target[cursor]['Start'], target[cursor]['End'], target[cursor]['Type'], target[cursor]['Score'], '"' + target[cursor]['Surface'].strip('"') + '"']) + '\n')
                    cursor += 1
