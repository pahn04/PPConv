gpu=0,1,2,3
model=ppcnnpp_scannet
exp_name=ppcnnpp_scannet
batch_size=256

mkdir -p logs_test

nohup python -u tools/test_scannet.py \
    --gpu ${gpu} \
    --model ${model}\
    --batch_size ${batch_size} \
    --with_rgb \
    --with_norm \
    --weight_dir ./pretrained/ppcnnpp_fp_scannet_caf.pth \
    >> logs_test/${exp_name}.log 2>&1 &

