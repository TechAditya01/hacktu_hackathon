# Define paths and parameters
$modelName = "model"
$version = "1.0"
$modelFile = "architecture.py"  # Path to model architecture file (if needed)
$serializedFile = "loan_approval_model.pth"  # Path to serialized model file
$handlerFile = "handler.py"  # Path to handler script
$exportPath = "./"  # Directory to save the .mar file

# Run torch-model-archiver
torch-model-archiver `
  --model-name $modelName `
  --version $version `
  --model-file $modelFile `
  --serialized-file $serializedFile `
  --handler $handlerFile `
  --export-path $exportPath `
  --force
