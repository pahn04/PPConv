gpu=0,1,2,3
model=ppcnnpp_s3dis
exp_name=ppcnnpp_s3dis
batch_size=256

mkdir -p logs_test

nohup python -u tools/test_s3dis.py \
    --gpu ${gpu} \
    --model ${model}\
    --batch_size ${batch_size} \
    --weight_dir ./pretrained/ppcnnpp_fp_s3dis_iwf.pth \
    --num_workers 0 --test_area 5 \
    >> logs_test/${exp_name}.log 2>&1 &

