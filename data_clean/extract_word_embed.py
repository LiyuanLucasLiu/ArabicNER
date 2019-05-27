import gensim
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default="./embed/full_uni_sg_100_wiki.mdl")
    parser.add_argument('-o', '--output', default="./embed/embed.txt")

    args = parser.parse_args()

    mod = gensim.models.Word2Vec.load(args.input)
    mod.wv.save_word2vec_format(args.output)
