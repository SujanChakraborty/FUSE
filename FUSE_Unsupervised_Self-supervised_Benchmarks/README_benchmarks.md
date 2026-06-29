Steps:
1. The file benchmarking_utils.py needs to be run first as it contains the logic of the unsupervised, self-supervised benchmarks along with the necessary functions to be loaded.
2. benchmarking_runner.ipynb generates the embeddings, saves them in benchmark_outputs folders and computes the classification results.
3. The averaged results across the datasets and the dataset wise results will be saved after running aggregation.ipynb.