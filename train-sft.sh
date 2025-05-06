#!/bin/bash

# Define variables

export WANDB_PROJECT="NLG-Project"
MODEL_NAME="EleutherAI/gpt-neo-1.3B"
DATASET_NAME="/home/kareem.elzeky/NLG/Project/data/mintaka"
LEARNING_RATE=2.0e-5
NUM_EPOCHS=5
TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION=2
LOGGING_STEPS=25
EVAL_DATASET="validation"
EVAL_STRATEGY="steps"
EVAL_STEPS=100
SAVE_STEPS=2000
OUTPUT_DIR="gpt-neo-SFT-5epochs"
REPORT_TO="wandb"
RUN_NAME="gpt-neo-SFT-2e-5-5epochs"
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