# 2026-05-11 02:30 UTC LB 認識改め

> ユーザー指示 (2026-05-11 02:30 UTC)「今のリーダーボード見てどれだけスコア改善しないといけないのか認識改めてね」への正面回答。LB は競合が改善し続けている = 我々の目標も上方修正必要。

## 1. 24h で公開 LB 動向

| 順位 | 5/10 旧 | **5/11 02:30 UTC 現** | delta |
|---|---|---|---|
| **Top 1 Virtute** | 9.256 | **9.132** | **-0.124** ⚠️ |
| Top 2 Silogram | 9.301 | 9.301 | 0 |
| Top 3 Mr.キノコ | 9.346 | 9.346 | 0 |
| **Top 4 賞金 cutoff anshul3501** | 9.415 | **9.374** | -0.041 |
| Top 5 Takahiro Saito | 9.584 | 9.498 | -0.086 |
| Top 20 Gold cutoff | spbforce 9.919 | Alhasan Abdellatif **9.865** | -0.054 |

= 競合 24h で **Top 1 が -0.124 ft 改善**。残 86 日で Top 1 が **8.5 帯まで降りる可能性大**。

## 2. 我々の現状 best LB **10.203** からの改善必要量

| target | 必要 ft |
|---|---|
| Top 20 Gold (= 9.865) | **-0.338** |
| Top 4 賞金圏 (= 9.374) | **-0.829** |
| **Top 1 (= 9.132)** | **-1.071** |
| 安全 Top 1 (= 残 86 日で 8.5 想定) | **-1.7** |
| **private 安全圏 (= shake-up 耐性 8.0-8.5)** | **-1.7〜-2.2** |

## 3. verified data からの conservative 評価

| case | 累積改善 | LB | 順位 |
|---|---|---|---|
| **best** (全 layer 上限) | -2.5 ft | **7.7** | **Top 1 圏** |
| **mid** (= 50% hit) | -1.4 ft | 8.8 | Top 10 |
| **conservative** (= 下限 + Edge R failure) | **-0.4 ft** | **9.8** | **Gold ぎりぎり** |
| worst (= 案 D/E も悪化) | +0.2 ft | 10.4 | 圏外 |

### verified layer

| layer | 期待 | verified |
|---|---|---|
| Edge S 単独 | -0.05〜-0.30 | ✅ **-0.114** (中域) |
| Edge R | -0.37 公開実測 | ⚠️ **+0.184 悪化** (refute) |
| 自前 4 base + Ridge 9-base | (改善想定) | ⚠️ **+0.360 悪化** (exp007 verify) |
| stratified fold + AV (= exp010) | -0.5〜-0.7 | smoke verify、実 LB 未 |
| 案 D Kalman / 案 E GP / Edge O | -0.3〜-0.8 | smoke pass、実 LB 未 |
| PySR / Mamba / TimeXer | -0.1〜-1.2 | spec 中、実 LB 未 |

= **verified 確実改善 = Edge S 単独 -0.114 のみ**、それ以外は推測。

## 4. 戦略 refine (= conservative case 防御)

1. **safe path 維持**: 5/12 plan で **1 sub を必ず safe (= exp005 v2 path Edge S 単独 10.203)** に reserve、shake-up 耐性確保
2. **AB ablation 厳格化**: 新 layer は単独 sub で必ず control 並走 (= Edge R 失敗の教訓)
3. **conservative case 対応**: exp010 が想定下回りなら exp011-013 で base 改造を全 skip し **Edge S 単独 + 軽 inject** に縮退
4. **競合 trajectory monitor**: 24h 毎 LB top 20 を確認、competitor の改善 trend を tracking

## 5. 5/12 sub plan **conservative refine**

| # | sub | mid case | conservative case |
|---|---|---|---|
| 1 | **exp010** (stratified fold + AV) | LB 9.5-9.7 | LB 10.0-10.2 (= 想定下回り、ただし悪化なし) |
| 2 | exp011 (= exp010 + Edge S) | 9.4-9.6 | 9.9-10.1 |
| 3 | exp012 (= + 案 D Kalman) | 8.5-8.9 | **10.2-10.5 (= 悪化)** ← rollback 判断 trigger |
| 4 | **exp013** (= + 案 E GP + Edge O) | **8.0-8.5 (= 9 切り)** | 9.5-9.7 |
| 5 | exp014a (= PySR) or exp014b (Edge R w_R=0.15 AB) | 7.7-8.2 | 9.4-9.6 |

= **conservative case でも Gold 圏到達**、ただし **9 切り達成は mid 以上必要**。

## 6. 中央 standing order (= 強化)

- 「軽量基準でなく優勝寄与最大化」 + 「idle 0」 + 「random sub に頼らず AB ablation」
- **競合 LB を 24h 毎に monitor、必要なら戦略修正**
- conservative case を常に backup、safe path 維持

## 7. 関連 doc

- `docs/dev/leaderboard.dense.md` — Submission Log (= intentional revert され古い state、touch 禁止)
- `docs/dev/submission-postmortems.dense.md` — postmortem
- `docs/dev/2026-05-11-state-snapshot.dense.md` — state snapshot
- `docs/dev/2026-05-11-h10-postmortem-and-fold-reform.dense.md` — H10
- `docs/dev/2026-05-11-final-synthesis-and-5-12-freeze.dense.md` — 5/12 plan freeze
- **本 doc** — LB 認識改め (= 5/12 final freeze の constraint)
