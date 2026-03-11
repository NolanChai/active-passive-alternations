#!/bin/bash

MODEL_NAME=$1
INPUT_DIR=$2
OUTPUT_DIR=$3

echo $MODEL_NAME
echo $INPUT_DIR
echo $OUTPUT_DIR

# .venv/bin/python run_uid_pipeline.py $INPUT_DIR $MODEL_NAME --uid_unit='word' --uid_level='document' --generate_counterfactual --output_dir $OUTPUT_DIR --output_name 'cf_word_document_uid.csv'
# echo 'Completed word-document experiment.'
# echo '=================================================================================================================================='
# .venv/bin/python run_uid_pipeline.py $INPUT_DIR $MODEL_NAME --uid_unit='token' --uid_level='document' --generate_counterfactual --output_dir $OUTPUT_DIR --output_name 'cf_token_document_uid.csv'
# echo 'Completed token-document experiment.'
# echo '=================================================================================================================================='
.venv/bin/python run_uid_pipeline.py $INPUT_DIR $MODEL_NAME --uid_unit='word' --uid_level='sentence' --generate_counterfactual --output_dir $OUTPUT_DIR --output_name 'cf_word_sentence_uid.csv'
echo 'Completed word-sentence experiment.'
echo '=================================================================================================================================='
.venv/bin/python run_uid_pipeline.py $INPUT_DIR $MODEL_NAME --uid_unit='token' --uid_level='sentence' --generate_counterfactual --output_dir $OUTPUT_DIR --output_name 'cf_token_sentence_uid.csv'
echo 'Completed token-sentence experiment.'
echo '=================================================================================================================================='
.venv/bin/python run_uid_pipeline.py $INPUT_DIR $MODEL_NAME --uid_unit='word' --uid_level='(-10,+10)' --generate_counterfactual --output_dir $OUTPUT_DIR --output_name 'cf_word_(-10,+10)_uid.csv'
echo 'Completed word-(-10,+10) experiment.'
echo '=================================================================================================================================='
.venv/bin/python run_uid_pipeline.py $INPUT_DIR $MODEL_NAME --uid_unit='token' --uid_level='(-10,+10)' --generate_counterfactual --output_dir $OUTPUT_DIR --output_name 'cf_token_(-10,+10)_uid.csv'
echo 'Completed token-(-10,+10) experiment.'
echo '=================================================================================================================================='

echo 'Experiment complete.'