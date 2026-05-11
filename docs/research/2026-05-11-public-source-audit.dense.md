# 公開 source dataset audit (= 優勝路手法 E 素材確認)

> 起源: docs/dev/2026-05-11-cv-lb-correlation.md § 5.4 (= 未統合 host dataset 5 件 list)
> 目的: 各 dataset の **数理本質的価値** を audit、 「優勝本質性」 criterion で採用判定
> 適用 criterion: `~/projects/kaggle/CLAUDE.md` § 11 (= 数理本質 / 優勝寄与 / rule 耐性 / datapoint 価値)

---

## 1. audit 済 dataset (= 2026-05-11)

### 1.1 `akshankrithick/rogii-gold-top10-direction-source` (138 KB、 5/9 公開)

**内容**:
- `gold_top10_submission.csv` のみ (= 464 KB、 test wells に対する predictions)
- schema: `id, tvt` (= row 単位)
- source code / model artifacts なし

**例 (= test well 000d7d20 の予測値)**:
```
000d7d20_1442,11746.069818072147
000d7d20_1443,11746.156777143762
000d7d20_1444,11746.171364689839
000d7d20_1445,11746.230501516411
```

**判定**:
- ❌ **集約 source として不採用** (= 「優勝本質性」 §11.1 違反: 「公開 source 集約で +X」 = paradigm 1 transduction、 他人解法依存)
- ✅ **diagnostic data として採用候補** (= per-row 比較で「我々がどの row で大きく外しているか」 を pinpoint、 数理本質ある analysis)

**diagnostic 用途の sketch**:
```python
# 我々の exp008 v2 (= LB 9.957) submission vs Gold Top10 (= LB 9.728)
ours = pd.read_csv("submissions/exp008_v2.csv")
gold = pd.read_csv("gold_top10_submission.csv")
diff = ours.merge(gold, on="id", suffixes=("_ours", "_gold"))
diff["abs_err"] = (diff["tvt_ours"] - diff["tvt_gold"]).abs()
# 大きく外している top-N rows を抽出 → どの well / 何 ft 深さで失敗しているか
worst = diff.sort_values("abs_err", ascending=False).head(100)
```

これは「Gold Top10 と我々の delta」 = 数理的 valuable (= 「Top10 が見つけている signal を我々が見落としている箇所」 を pinpoint)。

---

## 2. 未 audit dataset (= 次タスクで個別 audit)

| dataset | size | last update | 推測内容 | audit priority |
|---|---|---|---|---|
| `thbdh5765/rogii-v4-aeroridge-train-cache` | 1.57 GB | 5/11 04:53 | AeroRidge 系 train feature cache (= 別 base 候補) | **高** (= 5/11 最新 + AeroRidge は別 paradigm のヒント) |
| `buchananliang/rogii-karnak-top2-public-artefacts` | 6.6 MB | 5/8 04:07 | Top2 (= karnakbaev 系) 解法 artifacts | **中** (= 既存 karnakbaev dataset と差分確認、 重複なら不要) |
| `alfaxadeyembe/rogii-lgbm-model-artifacts` | 1.2 MB | 5/9 12:49 | LGBM 訓練済 artifacts | **中** (= LGBM 系 pretrained 流用候補) |
| `nina2025/rogii-03` | 507 KB | 5/11 06:19 | 小規模 utility (= 詳細不明) | 低 |
| `nina2025/rogii-07` | 1.1 MB | 5/7 22:26 | 別 version | 低 |
| `iathar/rogii-wellbore-models` | 2.2 MB | 5/7 12:46 | wellbore models | 低 |
| `geeknik/rogii-blend-public` | 71 KB | 5/8 01:36 | blend predictions | 低 (= 同様 submission-only の可能性) |
| `enisteper1/rogii-train-dataset` | 937 MB | 5/9 07:30 | 大容量、 詳細不明 | 中 |
| `innerf1re/rogii-blend-srcs-public` | 430 KB | 5/10 08:54 | blend sources | 中 |

---

## 3. audit プロセス (= 次タスク proceduralization)

各 dataset に対し:
1. `kaggle datasets download <slug> -p .work/source_audit/<short>/` で download
2. `unzip` + `ls -la` + 各 file の type 確認 (= source code / parquet / submission / model artifacts)
3. file 内容を 5-10 行 head で確認
4. 数理本質 判定:
   - source code あり = paradigm 1 でも valuable (= 解法 reverse engineer 可能)
   - submission only = transduction、 diagnostic 用途のみ
   - parquet (= train feature cache) = base model 改善の素材
   - model artifacts = 直接 blend に使用可能、 ただし licence 確認必須
5. 「優勝本質性」 criterion で採用判定
6. 本 doc § 1 に audit entry 追加

---

## 4. 「優勝本質性」 で篩った後の戦略

優勝本質性 §11.1 の binary table:

| ❌ 採るな | ✅ 採れ |
|---|---|
| "他の人がやってるから" | "他の人がやってないが理論で優位" |
| "公開 source 集約で +X 取れるから" | "self-compiled で独自 lift" |
| "コードが書きやすいから" | "数理本質で問題を解く" |

= **submission-only dataset は不採用 (= transduction paradigm)**、 source code / feature cache / model artifacts は **数理本質を理解できる場合のみ採用**。

逆に重要:
- **diagnostic data として常に valuable** (= Top10 submission との per-row diff は「自分の error 分布」 を data 由来で確認可能)
- これは「自前 self-compile」 の方向性決定の入力

---

## 5. 関連

- `docs/dev/2026-05-11-cv-lb-correlation.md` § 5.4 = 元 host dataset list
- `~/projects/kaggle/CLAUDE.md` § 11 = 優勝本質性 criterion
- `~/projects/kaggle/CLAUDE.md` § 4 = 4 paradigm 強制ルール (= paradigm 1 単独 では gold 不可)
- 別タスク draft: `docs/research/2026-05-11-karnakbaev-refit-pipeline-draft.dense.md` (= paradigm 2 拡張)
- 別タスク draft: `docs/research/2026-05-11-winning-path-A-nn-sketch.dense.md` (= paradigm 2 sequence model)
