if [ ! -d "rabbit" ]; then
    mkdir rabbit
fi

for dataset in ogbn-products ; do
    path=rabbit/$dataset-edge-list.txt
    if [ ! -f "$path" ]; then
        python -u exp/baseline/data_processing.py --dataset $dataset >> $path
    fi
done

echo "Generate edge list done."

cd rabbit
if [ ! -d "rabbit_order" ]; then
    git clone https://github.com/araij/rabbit_order.git
fi

cd rabbit_order/demo
make

for dataset in ogbn-products ; do
    path=~/inference_helper/rabbit/$dataset-rabbit-order.txt
    if [ ! -f "$path" ]; then
        reorder ~/inference_helper/rabbit/$dataset-edge-list.txt >> $path
    fi
done
