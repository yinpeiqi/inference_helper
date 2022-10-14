
python3 -u exp/baseline/run.py --model GCN --use-uva --dataset livejournal1 --auto --num-layers 3 --reorder
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset livejournal1 --auto --num-layers 3 --reorder
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset livejournal1 --auto --num-layers 2 --reorder

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset friendster --auto --num-layers 3 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset friendster --auto --num-layers 3 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset friendster --auto --num-layers 2 --reorder --free-rate 0.8

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset friendster --auto --num-layers 2 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset friendster --auto --num-layers 2 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset friendster --auto --num-layers 1 --reorder --free-rate 0.8


python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-papers100M --auto --num-layers 3 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset ogbn-papers100M --auto --num-layers 3 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset ogbn-papers100M --auto --num-layers 2 --reorder --free-rate 0.8

python3 -u exp/baseline/run.py --model GCN --use-uva --dataset ogbn-papers100M --auto --num-layers 2 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model GAT --use-uva --dataset ogbn-papers100M --auto --num-layers 2 --reorder --free-rate 0.8
python3 -u exp/baseline/run.py --model JKNET --use-uva --dataset ogbn-papers100M --auto --num-layers 1 --reorder --free-rate 0.8
