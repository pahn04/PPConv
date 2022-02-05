gpu=
model=ppcnnpp_scannet
extra_tag=ppcnnpp_scannet
batch_size=256
epoch=

nohup python -u tools/test_scannet.py \
    --gpu ${gpu} \
    --model ${model}\
    --batch_size ${batch_size} \
    --with_rgb \
    --with_norm \
    --weight_dir logs_scannet/${extra_tag}/best_model.pth \
    >> test/${extra_tag}_epoch${epoch}_bs${batch_size}.log 2>&1 &

