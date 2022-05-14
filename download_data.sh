python create_feature_in_ssd.py

python exp/baseline/run.py --dataset ogbn-products --load-data --reorder
python exp/baseline/run.py --dataset ogbn-products --load-data


python exp/baseline/run.py --dataset ogbn-papers100M --load-data --reorder
python exp/baseline/run.py --dataset ogbn-papers100M --load-data

cd ..
mkdir dataset
cd dataset
wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gunzip https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
cd ../inference_helper

python exp/baseline/run.py --dataset friendster --load-data --reorder
python exp/baseline/run.py --dataset friendster --load-data
