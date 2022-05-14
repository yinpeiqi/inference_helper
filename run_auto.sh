for rate in 0.8 0.85 0.9;
do
    python -u exp/baseline/run.py --dataset ogbn-products --model GCN --reorder --ssd --auto --free-rate $rate
    python -u exp/baseline/run.py --dataset ogbn-products --model GAT --reorder --ssd --auto --free-rate $rate
    python -u exp/baseline/run.py --dataset ogbn-products --model JKNET --reorder --ssd --auto --free-rate $rate

    python -u exp/baseline/run.py --dataset friendster --model GCN --reorder --ssd --auto --free-rate $rate
    python -u exp/baseline/run.py --dataset friendster --model GAT --reorder --ssd --auto --free-rate $rate
    python -u exp/baseline/run.py --dataset friendster --model JKNET --reorder --ssd --auto --free-rate $rate

    python -u exp/baseline/run.py --dataset ogbn-papers100M --model GCN --reorder --ssd --auto --free-rate $rate
    python -u exp/baseline/run.py --dataset ogbn-papers100M --model GAT --reorder --ssd --auto --free-rate $rate
    python -u exp/baseline/run.py --dataset ogbn-papers100M --model JKNET --reorder --ssd --auto --free-rate $rate
done