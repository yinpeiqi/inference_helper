
for rate in 0.85 ;
do
    for dataset in ogbn-papers100M friendster ogbn-products;
    do
        for model in GCN GAT JKNET;
        do
            # python -u exp/baseline/run.py --dataset $dataset --model $model --ssd --auto --free-rate $rate --debug
            python -u exp/baseline/run.py --dataset $dataset --model $model --reorder --ssd --auto --free-rate $rate --debug
            # python -u exp/baseline/run.py --dataset $dataset --model $model --reorder --ssd --batch-size 10000 --free-rate $rate --debug
            # python -u exp/baseline/run.py --dataset $dataset --model $model --reorder --ssd --reorder --batch-size 10000 --free-rate $rate --debug

        done
    done
done