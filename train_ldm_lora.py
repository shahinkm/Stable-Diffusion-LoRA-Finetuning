import sys

from src import parse_train_args, AutoMLPipeline

args = parse_train_args(sys.argv[1:])

train_pipeline = AutoMLPipeline.for_training(args)
train_pipeline.run()
