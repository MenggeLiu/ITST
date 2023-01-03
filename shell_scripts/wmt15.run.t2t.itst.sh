export CUDA_VISIBLE_DEVICES=3,4

data=~/simultaneous_translation/simul_confi/data_bins/deen_wmt15
modelfile=~/simultaneous_translation/ITST/checkpoints_wmt15/ITST_12-30

#data=PATH_TO_DATA
#modelfile=PATH_TO_SAVE_MODEL

python train.py --ddp-backend=no_c10d ${data} --arch transformer_itst \
 --optimizer adam \
 --max-update 200000 --patience 20 \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 2.5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --criterion label_smoothed_cross_entropy_with_itst_t2t \
 --label-smoothing 0.1 \
 --left-pad-source \
 --uni-encoder True \
 --no-progress-bar \
 --fp16 \
 --save-dir ${modelfile} \
 --max-tokens 4000 --update-freq 8 \
 --save-interval-updates 2000 \
 --keep-interval-updates 100 \
 --log-interval 100
