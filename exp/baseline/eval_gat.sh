#!/bin/bash
for model in GAT ;
    do
        for hidden in 256 ;
        do
        for head in 2 4 8 16 ;
            do
            python -u run.py --model $model --dataset reddit --auto --num-layers 3 --num-hidden $hidden --num-heads $head ;
            # for batch_size in 160000 80000 40000 20000 10000 5000 2500 ;
            for batch_size in 40000 20000 10000 5000 2500 1500;
                do
                python -u run.py --model $model --dataset reddit --batch-size $batch_size --num-layers 3 --num-hidden $hidden --num-heads $head ;
                done
            done
        done
    done
