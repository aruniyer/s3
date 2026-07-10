#!/usr/bin/env bash
set -euo pipefail

data_dir="${S3_DATA_DIR:-/data/ariy/s3-data}"

# Keep the large FAISS index in CPU RAM. Only the small E5 query encoder uses GPU 3.
export CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU:-3}"

python s3/search/retrieval_server.py \
    --index_path "$data_dir/e5_Flat.index" \
    --corpus_path "$data_dir/wiki-18.jsonl" \
    --topk 12 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --port 3000
