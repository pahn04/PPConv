gpu=0,1,2,3
model=ppcnnpp_s3dis
epoch=210
extra_tag=ppcnnpp_s3dis_best
batch_size=256
log_dir=logs_s3dis

nohup python -u tools/test_s3dis.py \
    --gpu ${gpu} \
    --model ${model}\
    --batch_size ${batch_size} \
    --weight_dir ./pretrained/ppcnnpp_s3dis_best.pth \
    --num_workers 0 --test_area 5 \
    >> test/${extra_tag}_epoch_${epoch}.log 2>&1 &

