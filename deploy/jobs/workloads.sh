export RESULTS_FILE=results/test_data_results.txt
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

export MODEL_REGISTRY=$(python -c "from src import config; print(config.MODEL_REGISTRY)")
aws s3 cp $MODEL_REGISTRY s3://madewithml/$GITHUB_USERNAME/mlflow/ --recursive
aws s3 cp results/ s3://madewithml/$GITHUB_USERNAME/results/ --recursive