python -u exp/baseline/run.py --dataset ogbn-products --model GCN --reorder --ssd --auto
python -u exp/baseline/run.py --dataset ogbn-products --model GAT --reorder --ssd --auto
python -u exp/baseline/run.py --dataset ogbn-products --model JKNET --reorder --ssd --auto

python -u exp/baseline/run.py --dataset friendster --model GCN --reorder --ssd --auto
python -u exp/baseline/run.py --dataset friendster --model GAT --reorder --ssd --auto
python -u exp/baseline/run.py --dataset friendster --model JKNET --reorder --ssd --auto

python -u exp/baseline/run.py --dataset ogbn-papers100M --model GCN --reorder --ssd --auto
python -u exp/baseline/run.py --dataset ogbn-papers100M --model GAT --reorder --ssd --auto
python -u exp/baseline/run.py --dataset ogbn-papers100M --model JKNET --reorder --ssd --auto
