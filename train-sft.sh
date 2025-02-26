#!/bin/bash

# Define variables

export WANDB_PROJECT="NLG-Project"
MODEL_NAME="openai-community/gpt2-large"
DATASET_NAME="PATH_TO_PREPROCESSED_DATASET"
LEARNING_RATE=2.0e-5
NUM_EPOCHS=15
TRAIN_BATCH_SIZE=64
GRADIENT_ACCUMULATION=1
LOGGING_STEPS=25
EVAL_DATASET="validation"
EVAL_STRATEGY="steps"
EVAL_STEPS=100
SAVE_STEPS=500
OUTPUT_DIR="gpt2-large-SFT-15batch"
REPORT_TO="wandb"
RUN_NAME="gpt2-large-SFT-2e-5-15batch"
PUSH_TO_HUB=False

# Run the training script
python ./sft.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --packing \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --gradient_checkpointing \
    --logging_steps "$LOGGING_STEPS" \
    --dataset_test_split "$EVAL_DATASET" \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" \
    --push_to_hub "$PUSH_TO_HUB"