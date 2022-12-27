export CUDA_VISIBLE_DEVICES=0
data=~/simultaneous_translation/simul_confi/data_bins/mustc_enzh
modelfile=~/simultaneous_translation/ITST/checkpoints_mustc/ITST_12-27
last_file=~/simultaneous_translation/ITST/checkpoints_mustc/ITST_12-27/checkpoint_last.pt
ref_dir=~/simultaneous_translation/simul_confi/data_raw/mustc_enzh/tst-COMMON.tok.zh
detok_ref_dir=~/simultaneous_translation/simul_confi/data_raw/mustc_enzh/tst-COMMON.zh
mosesdecoder=~/simultaneous_translation/mosesdecoder # https://github.com/moses-smt/mosesdecoder
src_lang=en
tgt_lang=zh


# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
    --output ${modelfile}/average-model.pt --last_file ${last_file}
file=${modelfile}/average-model.pt 


for threshold in 0.2 0.4 0.6 0.8 1.0; do
#threshold=TEST_THRESHOLD # test threshold in ITST, such as 0.8
# generate translation
  echo test threshold = $threshold
  python fairseq_cli/sim_generate.py ${data} --path ${file} \
      --batch-size 1 --beam 1 --left-pad-source --fp16 --remove-bpe \
      --itst-decoding --itst-test-threshold ${threshold} > pred.out 2>&1

  # latency
  echo -e "\nLatency"
  tail -n 4 pred.out

  # BLEU
  echo -e "\nBLEU"
  grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
  multi-bleu.perl -lc ${ref_dir} < pred.translation

  # SacreBLEU
  echo -e "\nSacreBLEU"
  perl ${mosesdecoder}/scripts/tokenizer/detokenizer.perl -l ${tgt_lang} < pred.translation > pred.translation.detok
  cat pred.translation.detok | sacrebleu ${detok_ref_dir} --w 2

done