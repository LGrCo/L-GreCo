workspace="./workspace_stats/"
mkdir -p $workspace

fairseq-train --task language_modeling data-bin/wikitext-103 \
 --save-dir checkpoints/transformer_wikitext-103/LGreCo32 --arch transformer_lm \
 --share-decoder-input-output-embed --dropout 0.1 --optimizer adam \
 --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 --lr 0.0005 \
  --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
   --tokens-per-sample 512 --sample-break-mode none --max-tokens 512 \
   --update-freq 16 --max-update 50000 2>&1 | tee $workspace/LGreCo32

fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/transformer_wikitext-103/LGreCo32/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400 2>&1 | tee $workspace/LGreCo32_val
