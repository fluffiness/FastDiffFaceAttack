export CUDA_VISIBLE_DEVICES=$1
python main_face.py --cross_attn_reg_weight 0
python main_face.py --cross_attn_reg_weight 6000
python main_face.py --cross_attn_reg_weight 9000
python main_face.py --cross_attn_reg_weight 12000
python ../k.py -id 0 -m 0.8 -o