set -e 

# export CUDA_VISIBLE_DEVICES=6

mustc_root=/home/liumengge/datasets_local/MuST-C/v2.0
lang=zh
modelfile=/home/liumengge/ITST/checkpoints_st/mustc_enzh

wav_list=/home/liumengge/datasets_local/MuST-C/v2.0/en-zh/eval/tst-COMMON/tst-COMMON.wav_list
reference=/home/liumengge/datasets_local/MuST-C/v2.0/en-zh/eval/tst-COMMON/tst-COMMON.zh

# test threshold in ITST, such as 0.8
threshold=0.95
gpu_id=6
port=1237
ckpt=checkpoint_best.pt
output_dir=/home/liumengge/ITST/simul_st_decodes/mustc_enzh_st/$ckpt/delta$threshold

# average best 5 checkpoints
# python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
#     --output ${modelfile}/average-model.pt --best True
file=${modelfile}/$ckpt

CUDA_VISIBLE_DEVICES=$gpu_id \
simuleval --agent ~/ITST/examples/speech_to_text/simultaneous_translation/agents/simul_agent.s2t.itst.flexible_predecision.py \
    --source ${wav_list} \
    --target ${reference} \
    --data-bin /home/liumengge/datasets_local/MuST-C/v2.0/en-zh \
    --config config_raw_joint.yaml \
    --model-path ${file} \
    --test-threshold ${threshold} \
    --lang ${lang} --sacrebleu-tokenizer zh \
    --output ${output_dir} \
    --scores --gpu \
    --port $port
