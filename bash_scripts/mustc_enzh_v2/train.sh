export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mustc_root=/home/liumengge/datasets_local/MuST-C/v2.0
lang=zh
modelfile=/home/liumengge/ITST/checkpoints_st/mustc_enzh_decay_step5w
wav2vec_path=/home/liumengge/ITST/wav2vec_ckpts/wav2vec_small.pt

# unidirectional Wav2Vec2.0 and unidirectional encoder
fairseq-train ${mustc_root}/en-${lang} \
  --config-yaml config_raw_joint.yaml \
  --train-subset train_joint \
  --valid-subset dev_raw_st \
  --save-dir ${modelfile} \
  --max-tokens 4800000  \
  --update-freq 2 \
  --task speech_to_text_wav2vec \
  --criterion label_smoothed_cross_entropy_with_itst_s2t_flexible_predecision \
  --report-accuracy \
  --arch convtransformer_espnet_base_wav2vec_itst \
  --w2v2-model-path ${wav2vec_path} \
  --uni-encoder True \
  --uni-wav2vec True \
  --optimizer adam \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --clip-norm 10.0 \
  --seed 1 \
  --ddp-backend=no_c10d \
  --keep-best-checkpoints 5 \
  --best-checkpoint-metric accuracy \
  --maximize-best-checkpoint-metric \
  --save-interval-updates 1000 \
  --keep-interval-updates 50 \
  --max-source-positions 800000 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
  --empty-cache-freq 1000 \
  --ignore-prefix-size 1 \
  --fp16 \
  --train-delta-min 0.5 --train-delta-decay-steps 50000 \
  # --reset-dataloader \
