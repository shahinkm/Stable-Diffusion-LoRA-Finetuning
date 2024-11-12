import sys

from src import parse_eval_args, AutoMLPipeline

args = parse_eval_args(sys.argv[1:])

eval_pipeline = AutoMLPipeline.for_evaluation(args)
eval_pipeline.run()
