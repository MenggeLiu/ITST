main_dir=~/simultaneous_translation/ITST
mustc_data_bin=~/simultaneous_translation/simul_confi/data_bins/deen_wmt15
mustc_data_raw=~/simultaneous_translation/simul_confi/data_raw/wmt15_ende/prep/

model_dir=~/simultaneous_translation/ITST/checkpoints_wmt15/ITST_12-30
test_name=test

#export PYTHONPATH=${main_dir}/src/simuleval/simuleval/agents
export CUDA_VISIBLE_DEVICES=1
for delta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
simuleval \
  --agent ${main_dir}/agents/simul_itst_t2t_enzh.py \
  --data-bin $mustc_data_bin \
  --model-path $model_dir/checkpoint_best.pt \
  --src_bpe_code $mustc_data_raw/bpe.32000.en-de \
  --sacrebleu-tokenizer 13a \
  --eval-latency-unit word \
  --aggressive_dash True \
  --output $main_dir/simul_decode/wmt15_deen/$test_name/result_${delta} \
  --stream $main_dir/simul_decode/wmt15_deen/$test_name/result_${delta}.de \
  --delta ${delta} \
  --source ${mustc_data_raw}/${test_name}.de \
  --target ${mustc_data_raw}/${test_name}.en \
  --gpu \
  --tgt_lang en \
  --port 12349
#cat result_${delta}.zh | sacrebleu ${mustc_data}/tst-COMMON.zh --w 2 --tokenize zh
done
