#!/bin/bash
INPUT_DIR="doc_amr_baseline/output_dataset_amr"
OUTPUT_DIR="doc_amr_baseline/output_doc_amr"

mkdir -p "$OUTPUT_DIR"

for dataset_path in "$INPUT_DIR"/*; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        echo "Processing dataset: $dataset_name"
        mkdir -p "$OUTPUT_DIR/$dataset_name"
        bash doc_amr_baseline/run_doc_amr_baseline.sh "$dataset_path" "$OUTPUT_DIR/$dataset_name" docAMR
    fi
done

echo "Generating AMR relations JSON..."
python doc_amr_baseline/generate_relation_vocab.py
