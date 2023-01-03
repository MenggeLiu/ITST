export CUDA_VISIBLE_DEVICES=1
data=~/simultaneous_translation/simul_confi/data_bins/deen_wmt15
modelfile=~/simultaneous_translation/ITST/checkpoints_wmt15/ITST_12-30
last_file=$modelfile/checkpoint_last.pt
ref_dir=~/simultaneous_translation/simul_confi/data_raw/wmt15_ende/prep/test.tok.en
detok_ref_dir=~/simultaneous_translation/simul_confi/data_raw/wmt15_ende/prep/test.tok.en
mosesdecoder=~/simultaneous_translation/mosesdecoder # https://github.com/moses-smt/mosesdecoder
src_lang=de
tgt_lang=en


# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 \
    --output ${modelfile}/average-model.pt --last_file ${last_file}
file=${modelfile}/average-model.pt 


for threshold in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#threshold=TEST_THRESHOLD # test threshold in ITST, such as 0.8
# generate translation
  echo test threshold = $threshold
  echo $data $file
  outdir=decodes/wmt15_deen_itst/threshold$threshold
  mkdir -p $outdir
  python fairseq_cli/sim_generate.py ${data} --path ${file} \
      --batch-size 1 --beam 1 --left-pad-source --fp16 --remove-bpe \
      --itst-decoding --itst-test-threshold ${threshold} > $outdir/pred.out 2>&1
  # latency
  echo -e "\nLatency"
  tail -n 4 $outdir/pred.out

  # BLEU
  echo -e "\nBLEU"
  grep ^H $outdir/pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > $outdir/pred.translation
  multi-bleu.perl -lc ${ref_dir} < $outdir/pred.translation

  # SacreBLEU
  echo -e "\nSacreBLEU"
  perl ${mosesdecoder}/scripts/tokenizer/detokenizer.perl -l ${tgt_lang} < $outdir/pred.translation > $outdir/pred.translation.detok
  cat $outdir/pred.translation.detok | sacrebleu ${detok_ref_dir} --w 2 -tok 13a

done
