main_dir=~/simultaneous_translation/ITST
mustc_data_raw=~/simultaneous_translation/simul_confi/data_raw/mustc_enzh
mustc_data_bin=~/simultaneous_translation/simul_confi/data_bins/mustc_enzh

model_dir=~/simultaneous_translation/ITST/checkpoints_mustc/ITST_12-27
test_name=tst-COMMON

#export PYTHONPATH=${main_dir}/src/simuleval/simuleval/agents
export CUDA_VISIBLE_DEVICES=1
for delta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
simuleval \
  --agent ${main_dir}/agents/simul_itst_t2t_enzh.py \
  --data-bin $mustc_data_bin \
  --model-path $model_dir/checkpoint_best.pt \
  --src_bpe_code $mustc_data_raw/bpe.30000.en \
  --sacrebleu-tokenizer zh \
  --eval-latency-unit char \
  --aggressive_dash True \
  --output $main_dir/simul_decode/mustc_enzh/$test_name/best_result_${delta} \
  --stream $main_dir/simul_decode/mustc_enzh/$test_name/best_result_${delta}.zh \
  --delta ${delta} \
  --source ${mustc_data_raw}/${test_name}.en \
  --target ${mustc_data_raw}/${test_name}.zh \
  --gpu \
  --tgt_lang zh \
  --port 12315
#cat result_${delta}.zh | sacrebleu ${mustc_data}/tst-COMMON.zh --w 2 --tokenize zh
done
