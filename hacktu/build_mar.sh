#!/bin/bash
# Ensure torch-model-archiver is installed: pip install torch-model-archiver
torch-model-archiver --model-name neural_model \
  --version 1.0 \
  --serialized-file loan_approval_model.pth \
  --handler handler.py \
  --extra-files "loan_approval_preprocessor.joblib,label_encoders.joblib,architecture.py" \
  --export-path model_store
