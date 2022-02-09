## Test Result
model: 2 layer GCN

hidden dim: 16

dataset: Reddit

Inference on GPU:
| dataloader: | GPU | CPU 0 worker | CPU 1 worker | CPU 4 worker |
|  ----  | ----  | ---- | ---- | ---- |
| dataloading  | 1.72, 1.50 | 6.76, 6.45| 1.66, 6.67 | 1.62, 2.20 |
| feature index | 7.48, 0.35 | 7.49, 0.36 | 9.56, 0.60 | 9.83, 0.75 |
| feature to GPU | 6.48, 0.13 | 6.83, 0.13 | 6.65, 0.13 | 7.22, 0.18 |
| graph to GPU| 0.30, 0.30 | 0.58, 0.57 | 0.78, 0.79 | 0.84, 1.03 |
| inference | 0.98, 0.77 | 1.02, 0.82 | 0.98, 0.85 | 1.03, 1.08 |
| feature to CPU | 0.05, 0.05 | 0.04, 0.04 | 0.09, 0.17 | 0.09, 0.21 |
| total | 20.12 | 31.11 | 29.22 | 26.41 |

Inference on CPU:
| dataloader: | GPU | CPU 0 worker | CPU 1 worker | CPU 4 worker |
|  ----  | ----  | ---- | ---- | ---- |
| dataloading | 7.06, 1.58 | 6.61, 6.41 | 1.95, 8.76 | 1.61, 2.36 |
| feature index | 7.25, 0.36 | 7.50, 0.34 | 9.91, 0.69 | 10.12, 0.81 |
| inference | 4.40, 0.60 | 4.84, 0.63 | 6.19, 1.01 | 6.41, 1.13 |
| total | 22.11 | 26.82 | 30.11 | 24.22 |

When dataloader device set GPU, the tensor of input node's index, graph, and output node's index is in GPU, reduce the 'graph to GPU' time. When worker use more workers, data will be loaded to different process and that cause the expand in 'feature index'.