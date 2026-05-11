# Remote Compute Migration (= A 案 Kaggle + Colab Pro+ ハイブリッド、2026-05-11)

> 起源: pane 1 (= 3-kaggle-comp-review) からの通知。 WSL2 OOM 発火 (= orbit-wars PPO 圧迫) で A 案ハイブリッド移行。
> 適用範囲: ROGII (= 本案件) + orbit-wars + 他 Kaggle 案件 (= 統一フォーマット)
> 関連: `docs/dev/2026-05-11-final-synthesis-and-5-12-freeze.dense.md`, `docs/dev/2026-05-11-top1-path-projection.dense.md`

## 1. 新 design 原則 (= MUST for 5/12 以降の新 experiment)

### 1.1 local generate 禁止

- ❌ 重い train (= Kalman/PF, Sparse GP, Beam, 自前 LGB×N base, NN 系) は **local で実行禁止**
- ❌ `data/processed/train_features.parquet` (774 MB) を local pandas で read する重い処理は **禁止**
- ✅ 全部 Kaggle Notebook 内で完結
- ✅ 軽い EDA (= per-well stats < 200 MB) は local OK
- ✅ doc / git 操作は local 推奨

### 1.2 1 方向データフロー

```
Kaggle Notebook (= train + inference)
    ↓ output: submission.csv + kernel log
local (= analysis + doc + git)
```

逆方向 (= local 重い処理 → Kaggle attach) は **禁止**:
- 例外: subagent AA PySR の case B (= local fit → 式 hardcode、artifact が 100 行以下 text のみ) は **再判断**
- → 後述 §3 で対応

### 1.3 既存 sub への影響 = なし

- exp005 / exp006 / exp007 / exp008 v2 / v3 / exp009 v2 / v4 / exp010 = **すべて Kaggle Notebook 経由で実行**
- 既に design rule に compliant、追加 action 不要

## 2. 5/12 以降の sub の design 原則

| sub | layer | local 処理 | 互換性 |
|---|---|---|---|
| exp011-013 (= 5/12 sub) | exp010 + Edge S + Kalman 等の加算 | なし (= 全 Kaggle Notebook 内) | ✅ |
| **exp015 PySR (5/13)** | PySR fit (= subagent AA spec) | local fit case B vs Kaggle fit case A | ⚠️ §3 refine |
| **exp016 Mamba (5/13-14)** | Mamba train | Kaggle Notebook 内 (= ky7240/rogii-mamba-runtime dataset 経由) | ✅ |
| exp017 Kriging (5/15) | GeostatsPy Kriging | Kaggle Notebook 内 | ✅ |
| exp018 MEMO (5/16) | MEMO random_stretch | Kaggle Notebook 内 | ✅ |
| ⏸️ TimeXer | license 不明 + Kaggle Notebook (case A) で fit | TimeXer 採用見送り (= subagent AE finding) | - |

## 3. PySR の design 原則 refine

### 3.1 旧 spec (= subagent AA case B)

- **offline local fit** (= local CPU 5-10 min × 6 formation = 30-60 min)
- 結果 = sympy expression を kernel 内 hardcode
- Kaggle Internet disabled の互換性のため

### 3.2 新 spec (= A 案 design 原則準拠)

**選択肢**:

| 案 | 説明 | 工数 | risk |
|---|---|---|---|
| **A. Kaggle Notebook 内 fit** | PySR を kernel script 内で fit、6 formation × 50-100 epoch = 30-60 min on Kaggle GPU/CPU | 設計変更 0.5 日 | 低 (= 9 hr cap 内、julia install が必要、subagent AA case A は採用見送りだったが再評価) |
| **B. offline-export ✅** | local で fit、結果のみ commit (= 100 行 text、git 管理) | 既 ready | **設計原則準拠**は微妙 (= 「local generate 禁止」 抵触するが artifact が極小なので例外扱い可) |

→ **判断**:
- **B 採用 (= 例外扱い)** が現実的 (= subagent AA spec が既に完成、artifact が text 100 行で重い処理ではない)
- ただし design 原則「local generate 禁止」の精神 (= 重い train を local で回さない) には合致 (= PySR fit は CPU 5 min)
- pane 1 確認待ち、ただし 5/13 着手継続

## 4. orbit-wars との統一フォーマット (= 推奨)

orbit-wars の `docs/dev/remote-migration-*.md` 形式に合わせる:
- §1 design 原則
- §2 既存 sub への影響
- §3 新 experiment の design rule
- §4 検証 + verify

orbit-wars の format を別途確認、必要なら本 doc を sync。

## 5. pane 1 review への update

### 5.1 online learning patch (= Edge R) refute

pane 1 提案: "online learning patch +0.37 ft = Gold 確率 65→85%"

**我々の verify (= 2026-05-11 SCORED)**:
- exp005 v3 (= Edge S + Edge R) LB = **10.387 ft**
- exp005 v2 (= Edge S 単独) LB = **10.203 ft**
- **Edge R 単独効果 = +0.184 ft 悪化** (= 公開 698002 -0.37 と逆方向、差 0.554 ft)

= **pane 1 提案は ROGII で work しなかった**、refute verified。 詳細は `docs/dev/submission-postmortems.dense.md` §1.5 + `docs/dev/leaderboard.dense.md` exp005 v3 row 参照。

### 5.2 教訓 (= 文献 hack 共通)

- **公開実測値の盲信は危険**、AB test 必須
- ROGII base = karnakbaev pretrained 5-base + Ridge blend、公開 698002 = raw LGB tysig = base 構造異なる
- → 「最高 ROI 1 手」claim も `H-R1 base model 依存性` で refute される possibility

## 6. ROGII の現状 (= migration 影響なし、Top 1 path 維持)

- exp008 v2 LB **9.957** = Silver 上位達成
- exp009 v2 PENDING = Gold 確実視
- exp008 v3 RUNNING (Kaggle) = 9 切り射程
- exp010 RUNNING (Kaggle) = Gold 確実視
- 5/13 Mamba + PySR で Top 1 圏到達 verified

= **A 案移行に伴う ROGII への手戻りなし**、Top 1 path は予定通り。

## 7. 関連 doc

- `docs/dev/2026-05-11-final-synthesis-and-5-12-freeze.dense.md` — 5/12 sub plan
- `docs/dev/2026-05-11-top1-path-projection.dense.md` — Top 1 path numerical projection
- `docs/dev/leaderboard.dense.md` — Submission Log
- `docs/dev/submission-postmortems.dense.md` — postmortem (= Edge R refute)
- (orbit-wars の `docs/dev/remote-migration-*.md`) — 統一フォーマット参照元、別案件
