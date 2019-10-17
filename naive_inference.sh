green=`tput setaf 2`
reset=`tput sgr0`

INPUT_FILE=$1
OUTPUT_FOLDER="/wdata"
OUTPUT_FILE=$2
DEVICE=${3:-auto}

if [ ! -e $OUTPUT_FOLDER/sner0 ]; then
    echo ${green}=== Downloading Pre-trained Models ===${reset}
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-MHoxEgqTA8iu1jb42C_Zb_vmnI1i-7o' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-MHoxEgqTA8iu1jb42C_Zb_vmnI1i-7o" -O ${OUTPUT_FOLDER}/model.tar && rm -rf /tmp/cookies.txt

    echo ${green}=== Extracting Models ===${reset}
    tar -C ${OUTPUT_FOLDER} -xvf ${OUTPUT_FOLDER}/model.tar && rm -rf ${OUTPUT_FOLDER}/model.tar
fi

echo ${green}=== Data Pre-processing ===${reset}
python naive_pre_process.py --file_input ${INPUT_FILE} --file_output ${OUTPUT_FOLDER}/tmp_output.json --config ./config/sner0.json

echo ${green}=== Model Ensembling and Inferencing ===${reset}
python ensemble_ner.py -o ${OUTPUT_FOLDER}/naive_output.csv --cp_root ${OUTPUT_FOLDER} -i ${OUTPUT_FOLDER}/tmp_output.json --config_list ./config/sner0.json -n ${INPUT_FILE} --gpu ${DEVICE}
