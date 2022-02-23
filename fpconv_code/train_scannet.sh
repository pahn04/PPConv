gpu=0,1,2,3
batch_size=64
exp_name=ppcnnpp_scannet

mkdir -p logs_scannet/${exp_name}

nohup python -u tools/train_scannet.py \
    --model ppcnnpp_scannet \
    --batch_size ${batch_size} \
    --save_dir logs_scannet/${exp_name} \
    --num_points 8192 \
    --gpu ${gpu} \
    --with_rgb \
    --with_norm \
    --workers 32 \
    --epochs 3000 \
    --lr 0.01 \
    --decay_step_list 1000 1500 2000 2500 \
    --lr_decay 0.5 \
    --weight_decay 0.001 \
    --accum ${batch_size} \
    --augment \
    >> logs_scannet/${exp_name}/nohup.log 2>&1 &

