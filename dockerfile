FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

RUN pip install torch-scope ipdb gensim

COPY data_clean/ data_clean/
COPY abnlp/ abnlp/
COPY config/ config/
COPY dict/ dict/
COPY post_process/ post_process/
COPY train_ner.py train_ner.py
COPY pre_process_train.py pre_process_train.py
COPY pre_process_test.py pre_process_test.py
COPY ensemble_ner.py ensemble_ner.py
COPY train.sh train.sh
COPY test.sh test.sh

RUN  apt-get update \
  && apt-get install -y wget unzip vim

RUN  mkdir embed

RUN  wget https://archive.org/download/aravec_3_wiki/full_uni_sg_100_wiki.zip \
  && unzip full_uni_sg_100_wiki.zip -d embed \
  && python ./data_clean/extract_word_embed.py \
  && rm ./embed/full_uni_sg_100_wiki.* \
  && rm ./full_uni_sg_100_wiki.zip

# fastText vectors are distributed under the Creative Commons Attribution-Share-Alike License 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.vec.gz \
  && gunzip cc.ar.300.vec.gz \
  && mv cc.ar.300.vec ./embed/cc.ar.300.vec

RUN wget https://www.dropbox.com/s/pu4njaye84gqipu/flm.th?dl=1 -O embed/flm.th
RUN wget https://www.dropbox.com/s/hrcjc7zrhpt4xyb/blm.th?dl=1 -O embed/blm.th
