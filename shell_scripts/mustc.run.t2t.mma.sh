export CUDA_VISIBLE_DEVICES=3

# latency weight in MMA
latency_weight=0.001

data=~/simultaneous_translation/simul_confi/data_bins/mustc_enzh
modelfile=~/simultaneous_translation/ITST/checkpoints_mustc/mma_l$latency_weight

#data=PATH_TO_DATA
#modelfile=PATH_TO_SAVE_MODEL
#


python train.py --ddp-backend=no_c10d ${data} --arch transformer_monotonic \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 2.5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 \
 --decoder-embed-dim 512 --decoder-ffn-embed-dim 1024 \
 --encoder-layers 6 --decoder-layers 6 \
 --encoder-attention-heads 4 \
 --decoder-attention-heads 4 \
 --criterion latency_augmented_label_smoothed_cross_entropy \
 --latency-weight-avg ${latency_weight} \
 --simul-type infinite_lookback \
 --label-smoothing 0.1 \
 --left-pad-source \
 --no-progress-bar \
 --find-unused-parameters \
 --fp16 \
 --save-dir ${modelfile} \
 --max-tokens 4000 --update-freq 8 \
 --save-interval-updates 1000 \
 --keep-interval-updates 200 \
 --log-interval 100
