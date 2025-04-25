#!/bin/env bash
conda activate himp-nc

seed=1
dataname=Cora
# ( Cora CiteSeer PubMed Cornell Texas Wisconsin Computers Photo Chameleon Squirrel Roman-empire Amazon-ratings Minesweeper Tolokers Questions )
gnn=GCN
# ( GCN GAT GIN GraphSAGE )
het_mode=heterophily
# ( original heterophily homophily mix )

save_dir=results_nc/${dataname}_${gnn}/seed_${seed}

python main.py --seed_split ${seed} --seed ${seed} --data_name ${dataname} --model_name ${gnn} --het_mode ${het_mode} --save_dir ${save_dir} \
        --max_epochs 2000 --num_layers 2 --hidden_channels 128 --learning_rate 1e-3 --dropout 0.2 --weight_decay 1e-5 \