#!/usr/bin/env bash
# Download PubMedQA PQA-L and the official test split; verify checksums.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
curl -sL -o pqa_labeled.parquet "https://huggingface.co/api/datasets/qiaojin/PubMedQA/parquet/pqa_labeled/train/0.parquet"
curl -sL -o test_ground_truth.json "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
shasum -a 256 -c <<SUMS
3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c  pqa_labeled.parquet
939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51  test_ground_truth.json
SUMS
