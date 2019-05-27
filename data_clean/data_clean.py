import argparse
import codecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_input', default="./config/ner0.json")
    parser.add_argument('--train_file_output', default="./config/ner0.json")
    args = parser.parse_args()

    with codecs.open(args.train_file_input, 'r', 'utf-8') as fin:
        lines = fin.readlines()

    punc_list = ['(', ':', ')', '،', '...', '.', ').']
    flag = False
    for idx in range(len(lines)):
        line = lines[idx]
        sl = line.split()
        if len(sl) > 1 and sl[0] in punc_list and sl[1] != 'O':
                lines[idx] = sl[0] + ' ' + 'O\n'
                flag = True
        elif flag:
            if 'I-' in sl[1]:
                sl[1] = sl[1].replace('I-', 'B-')
                lines[idx] = sl[0] + ' ' + sl[1] + '\n'
            flag = False
        else:
            flag = False

    Company_list = ['hp', 'ibm', 'microsoft', 'google', 'intel', 'cisco', 'oracle', 'qualcomm', 'emc', 'xerox', 'danaher', 'ebay']

    for idx in range(len(lines)):
        line = lines[idx]
        sl = line.split()
        if len(sl) > 1 and sl[0].lower() in Company_list and sl[1] == 'O':
            lines[idx] = sl[0] + ' ' + 'B-MIS\n'

    Cross_Validation_MIS_list_one_word = ['cde', 'gpl', 'shell', 'compiler', 'fifa', 'icccm']

    for idx in range(len(lines)):
        line = lines[idx]
        sl = line.split()
        if len(sl) > 1 and sl[0].lower() in Cross_Validation_MIS_list_one_word and sl[1] == 'O':
            lines[idx] = sl[0] + ' ' + 'B-MIS\n'

    Cross_Validation_MIS_list_two_word = ['spontaneous emission', 'photoelectric effect']

    for idx in range(len(lines)-1):
        sl = lines[idx].split()
        if len(sl) > 1 and sl[1] == 'O':
            nsl = lines[idx+1].split()
            if len(nsl) > 1 and nsl[1] == 'O' and sl[0].lower() + ' ' + nsl[0].lower() in Cross_Validation_MIS_list_two_word:
                lines[idx] = sl[0] + ' ' + 'B-MIS\n'
                lines[idx + 1] = nsl[0] + ' ' + 'I-MIS\n'

    Cross_Validation_MIS_list_three_word = ['common desktop environment', 'general public license']

    for idx in range(len(lines)-2):
        sl0 = lines[idx].split()
        sl1 = lines[idx+1].split()
        sl2 = lines[idx+2].split()
        if len(sl0) > 1 and sl0[1] == 'O' and len(sl1) > 1 and sl1[1] == 'O' and len(sl2) > 1 and sl2[1] == 'O':
            if ' '.join([sl0[0].lower(), sl1[0].lower(), sl2[0].lower()]) in Cross_Validation_MIS_list_three_word:
                lines[idx] = sl0[0] + ' ' + 'B-MIS\n'
                lines[idx + 1] = sl1[0] + ' ' + 'I-MIS\n'
                lines[idx + 2] = sl2[0] + ' ' + 'I-MIS\n'

    with codecs.open(args.train_file_output, 'w', 'utf-8') as fout:
        for line in lines:
            fout.write(line)
