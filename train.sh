green=`tput setaf 2`
reset=`tput sgr0`

INPUT_FOLDER=$1
OUTPUT_FOLDER="/wdata"

echo ${green}=== Removing Pre-trained Models ===${reset}
rm -rf ${OUTPUT_FOLDER}/*

echo ${green}=== Cleaning Data ===${reset}
python data_clean/data_clean.py --train_file_input ${INPUT_FOLDER}/train.txt --train_file_output ${OUTPUT_FOLDER}/train_cleaned.txt

echo ${green}=== Building Dictionary Based Model ===${reset}
python post_process/save_dictionary.py --train_file_input ${INPUT_FOLDER}/train.txt --dict_output ${OUTPUT_FOLDER}/train_dict.json

echo ${green}=== Data Pre-processing ===${reset}
python pre_process_train.py --train_file_input ${OUTPUT_FOLDER}/train_cleaned.txt --train_file_output ${OUTPUT_FOLDER}/train.json --config ./config/sner0.json

for ind in {0..8}
do
	echo ${green}=== Training Model ${ind} ===${reset}
	python train_ner.py --cp_root ${OUTPUT_FOLDER} --config ./config/sner${ind}.json --train_file ${OUTPUT_FOLDER}/train.json
done
