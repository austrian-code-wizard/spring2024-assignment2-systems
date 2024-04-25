for model_size in small medium large xl "2.7B"
do
    python cs336-systems/cs336_systems/benchmark.py --model-config $model_size "$@"
done