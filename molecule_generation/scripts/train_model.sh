#!/bin/bash
data=qm9
save_dir=results/test/${data}
qm9_paras="--b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 27 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 1"
zinc_paras="--b_n_flow 10  --b_hidden_ch 512,512  --a_n_flow 38  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2"

if [ $data == "qm9" ]; then
    MODEL_PARAS=$qm9_paras
fi
if [ $data == "zinc250k" ]; then
    MODEL_PARAS=$zinc_paras
fi

python hetflow/train_model.py \
 --data_name ${data} --save_dir=${save_dir} --data_dir=data \
  --batch_size 256  --max_epochs 500 --gpu 0  --debug True --save_epochs 10 --ite_log_step 75  \
  --record_all_metrics True  --true_adj True --graph_conv_option HetFlow --seed 0 ${MODEL_PARAS} 
#   --wandb_record True --project_name 'test_submission'