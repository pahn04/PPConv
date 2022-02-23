gpu=0,1,2,3
model=ppcnnpp_s3dis
exp_name=ppcnnpp_conv_pillar_iwf
log_dir=logs_s3dis
batch_size=64

mkdir -p ${log_dir}/${exp_name}

nohup python -u tools/train_s3dis.py \
    --save_dir ${log_dir}/${exp_name} \
    --model ${model} \
    --batch_size ${batch_size} \
    --gpu ${gpu} \
    >> ${log_dir}/${exp_name}/nohup.log 2>&1 &

