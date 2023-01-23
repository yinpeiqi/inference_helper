# Table 4
python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 1
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 1

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 2
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 2
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 1

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 3
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 3
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 2

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 4
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 4
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset ogbn-products --batch-size 200000 --num-layers 3

# Table 3
python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-products --batch-size 200000 --fan-out --num-layers 3
python3 -u exp/baseline/run.py --model GCN --use-uva --dataset livejournal1 --batch-size 100000 --fan-out --num-layers 3
python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-papers100M --batch-size 100000 --fan-out --num-layers 2
python3 -u exp/baseline/run.py --model GCN --use-uva --dataset friendster --batch-size 10000 --fan-out --num-layers 2

# Table 2
python3 -u exp/baseline/run.py --model GCN --use-uva --dataset livejournal1 --batch-size 100000 --num-layers 3
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset livejournal1 --batch-size 100000 --num-layers 3
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset livejournal1 --batch-size 100000 --num-layers 2

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-papers100M --batch-size 100000 --num-layers 2
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset ogbn-papers100M --batch-size 100000 --num-layers 2
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset ogbn-papers100M --batch-size 100000 --num-layers 1

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset friendster --batch-size 10000 --num-layers 2
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset friendster --batch-size 10000 --num-layers 2
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset friendster --batch-size 10000 --num-layers 1

python3 exp/baseline/hetero/main.py --model RGCN
python3 exp/baseline/hetero/main.py --model HGT

python3 exp/baseline/hetero/main_240m.py --model RGCN
python3 exp/baseline/hetero/main_240m.py --model HGT


