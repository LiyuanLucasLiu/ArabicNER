import argparse
import codecs
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_input', default="./config/ner0.json")
    parser.add_argument('--dict_output', default="./config/ner0.json")
    args = parser.parse_args()

    entity_list = list()

    surface_name = list()
    current = None

    with codecs.open(args.train_file_input, 'r', 'utf-8') as fin:
        for line in fin:

            if line and not line.isspace():
                word, label = line.split()

                if label.startswith('I-') and current is not None and label[2:] == current:
                    surface_name.append(word)
                else:
                    if current is not None:
                        entity_list.append([surface_name,current])

                    if label == 'O':
                        current = None
                        surface_name = list()
                    else:
                        current = label[2:]
                        surface_name = [word]

            else:
                if current is not None:
                    entity_list.append([surface_name,current])

                current = None
                surface_name = list()

        if current is not None:
            entity_list.append([surface_name,current])

    with open(args.dict_output, 'w') as fout:
        json.dump(entity_list, fout)
