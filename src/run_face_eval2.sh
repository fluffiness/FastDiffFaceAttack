export CUDA_VISIBLE_DEVICES=$1
python eval_face.py --ckpt_base_dir ../logs/ckpt_official/irse50_ca
python eval_face.py --ckpt_base_dir ../logs/ckpt_official/facenet_ca
python ../k.py -id 0 -m 0.7 -o