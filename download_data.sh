# python3 create_feature_in_ssd.py
# echo "create feature done!"

# # python3 exp/baseline/run.py --dataset ogbn-products --load-data --reorder
python3 exp/baseline/run.py --dataset ogbn-products --load-data
echo "finished ogbn-products"

cd ..
mkdir dataset
cd dataset
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
gunzip soc-LiveJournal1.txt.gz
cd ../inference_helper

# # python3 exp/baseline/run.py --dataset ogbn-papers100M --load-data --reorder
python3 exp/baseline/run.py --dataset ogbn-papers100M --load-data
echo "finished ogbn-papers100M"

cd ..
mkdir dataset
cd dataset
wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gunzip com-friendster.ungraph.txt.gz
cd ../inference_helper

# python3 exp/baseline/run.py --dataset friendster --load-data --reorder
python3 exp/baseline/run.py --dataset friendster --load-data
echo "finished friendster"
