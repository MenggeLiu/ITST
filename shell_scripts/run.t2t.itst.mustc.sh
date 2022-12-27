export CUDA_VISIBLE_DEVICES=0,1,2,3

data=~/simultaneous_translation/simul_confi/data_bins/mustc_${src}${tgt}
modelfile=~/simultaneous_translation/ITST/checkpoints_mustc/ITST_12-27

#data=PATH_TO_DATA
#modelfile=PATH_TO_SAVE_MODEL

python train.py --ddp-backend=no_c10d ${data} --arch transformer_itst \
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
 --encoder-layers 6 --decoer-layers 6 \
 --encoder-attention-heads 4 \
 --decoder-attention-heads 4 \
 --criterion label_smoothed_cross_entropy_with_itst_t2t \
 --label-smoothing 0.1 \
 --left-pad-source \
 --uni-encoder True \
 --no-progress-bar \
 --fp16 \
 --save-dir ${modelfile} \
 --max-tokens 8192 --update-freq 1 \
 --save-interval-updates 1000 \
 --keep-interval-updates 200 \
 --log-interval 100