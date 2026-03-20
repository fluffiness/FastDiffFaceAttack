export CUDA_VISIBLE_DEVICES=$2
python $1
python ../k.py -id 0 -m 0.7 -o
