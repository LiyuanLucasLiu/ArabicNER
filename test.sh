green=`tput setaf 2`
reset=`tput sgr0`

INPUT_FOLDER=$1
OUTPUT_FOLDER="/wdata"
OUTPUT_FILE=$2

if [ ! -e $OUTPUT_FOLDER/sner0 ]; then
    echo ${green}=== Downloading Pre-trained Models ===${reset}
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-MHoxEgqTA8iu1jb42C_Zb_vmnI1i-7o' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-MHoxEgqTA8iu1jb42C_Zb_vmnI1i-7o" -O ${OUTPUT_FOLDER}/model.tar && rm -rf /tmp/cookies.txt

    echo ${green}=== Extracting Models ===${reset}
    tar -C ${OUTPUT_FOLDER} -xvf ${OUTPUT_FOLDER}/model.tar && rm -rf ${OUTPUT_FOLDER}/model.tar
fi

echo ${green}=== Data Pre-processing ===${reset}
python pre_process_test.py --dev_file_input ${INPUT_FOLDER}/dev.txt --dev_file_output ${OUTPUT_FOLDER}/dev.json --test_file_input ${INPUT_FOLDER}/test.txt --test_file_output ${OUTPUT_FOLDER}/test.json --config ./config/sner0.json

echo ${green}=== Model Ensembling and Inferencing ===${reset}
python ensemble_ner.py -o ${OUTPUT_FOLDER}/tmp0.csv --cp_root ${OUTPUT_FOLDER} -i ${OUTPUT_FOLDER}/dev.json ${OUTPUT_FOLDER}/test.json

echo ${green}=== Dictionary Based Model Inferencing ===${reset}
python post_process/int_dictionary.py --dict_input ${OUTPUT_FOLDER}/train_dict.json --raw_dev ${INPUT_FOLDER}/dev.txt --raw_test ${INPUT_FOLDER}/test.txt --csv_output ${OUTPUT_FOLDER}/tmp1.csv

echo ${green}=== Results Merging ===${reset}
python post_process/merge_result.py --model_csv ${OUTPUT_FOLDER}/tmp0.csv --dict_csv ${OUTPUT_FOLDER}/tmp1.csv --csv_output ${OUTPUT_FILE}
