export CUDA_VISIBLE_DEVICES=$1
python main_face.py
python ../k.py -id 0 -m 0.7 -o
