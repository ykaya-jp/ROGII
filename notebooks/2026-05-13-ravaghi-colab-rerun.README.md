# Colab で ravaghi LB 9.43 fork 再現 (= plan §8.2、 Track 2.1)

## 使い方

1. **Colab Pro+ で notebook を開く**: `2026-05-13-ravaghi-colab-rerun.ipynb` を Colab に upload
2. **Runtime 設定**: Runtime > Change runtime type > Hardware accelerator: **A100 GPU**
3. **Cell 1 (= setup) を Run**: 
   - kaggle.json upload を促されたら、 local の `~/.kaggle/kaggle.json` を upload
   - dataset DL (= 2.4 GB) で約 5 min 待ち
4. **All cells を Run All**: 推定 30-60 min (= A100 で LGB/CB cache load + Climber + Optuna 500-trial)
5. **submission.csv が `/content/submission.csv` に出力**
6. **Last cell の instruction に従い Kaggle dataset として upload** = `ky7240/rogii-ravaghi-colab-output`

## 次の step (= local 戻り後)

```bash
# kaggle_kernels/exp019_ravaghi_colab_run/ を作成 (= dataset から submission.csv 読んで output)
# push + competition_submit_cli で submit
# LB 9.43 再現確認
```

## Expected
- LB **9.43 帯** (= ravaghi original 同等) → defense ライン確保
- LB > 9.7 (= 大破綻) → exp017 と同じ真因確認、 PF non-determinism 確定、 postmortem 起動
