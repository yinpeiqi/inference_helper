
layer=3
hidden=128
models=(GAT GCN JKNET)
datasets=(ogbn-products friendster ogbn-papers100M)
best_batches=(250000 20000 60000 500000 60000 250000 280000 15000 40000)

for i in {0..2}
    do
    for j in {0..2}
        do
        model=${models[i]}
        dataset=${datasets[j]}
        batch_size=${best_batches[i*3+j]}

        # auto solotion
        python -u exp/baseline/run.py --model $model --num-layers $layer --num-hidden $hidden --num-heads 2 --use-uva --dataset $dataset --auto

        # best batch baseline
        python -u exp/baseline/run.py --model $model --num-layers $layer --num-hidden $hidden --num-heads 2 --use-uva --dataset $dataset --batch-size $batch_size

    done
done
