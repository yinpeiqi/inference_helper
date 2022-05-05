
layer=3
hidden=128
rates=(0.7 0.9 0.9)
models=(GAT GCN JKNET)
datasets=(ogbn-products friendster ogbn-papers100M)

best_ran_batches=(250000 20000 60000 500000 60000 250000 280000 30000 40000)
best_rcmk_batches=(250000 10000 20000 500000 40000 130000 280000 8000 25000)

for i in {0..2}
    do
    for j in {0..2}
        do
        rate=${rates[i]}
        model=${models[i]}
        dataset=${datasets[j]}
        ran_batch_size=${best_ran_batches[i*3+j]}
        rcmk_batch_size=${best_rcmk_batches[i*3+j]}

        # auto solotion
        python -u exp/baseline/run.py --model $model --num-layers $layer --num-hidden $hidden --num-heads 2 --use-uva --dataset $dataset --auto --free-rate $rate

        # auto solotion with reorder
        python -u exp/baseline/run.py --model $model --num-layers $layer --num-hidden $hidden --num-heads 2 --use-uva --dataset $dataset --auto --reorder --free-rate $rate

        # best batch baseline
        python -u exp/baseline/run.py --model $model --num-layers $layer --num-hidden $hidden --num-heads 2 --use-uva --dataset $dataset --batch-size $ran_batch_size

        # best batch baseline with reorder
        python -u exp/baseline/run.py --model $model --num-layers $layer --num-hidden $hidden --num-heads 2 --use-uva --dataset $dataset --batch-size $rcmk_batch_size --reorder

    done
done