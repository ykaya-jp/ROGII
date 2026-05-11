# exp010 Push Debug (2026-05-11)

> branch: `feat/phase-6-exp010-fold-reform`
> 担当: subagent AB
> 結論: **slug rename** (`rogii-exp010-fold-reform` → `rogii-exp010-foldreform-v1`) で push 成功。kernel script は logical 変更ゼロ。

---

## 0. 序: 「Notebook not found」 root cause 調査

### 0.1 観測されていた症状
- `kaggle kernels push -p kaggle_kernels/exp010_fold_reform` で `Kernel push error: Notebook not found`
- 中央が試行した 2 slug 両方で再現:
  - `ky7240/rogii-exp010-fold-reform` (= original)
  - `ky7240/rogii-exp010-fr` (= 短縮 slug)
- 同 branch の **exp008 case D kalman** は同一 dataset_sources / 同一 GPU 設定で **push 成功済** (= 5/10 22:53 SCORED)
- 別 error 「Maximum batch GPU session count of 2 reached」も交互発生 (= 同時 2 GPU 上限の Kaggle quota)

### 0.2 確認した仮説 (= 棄却したもの)
| # | 仮説 | 検証手段 | 結論 |
|---|---|---|---|
| H1 | kernel script に syntax error | `py_compile` | 棄却: `syntax OK` |
| H2 | kernel size 超過 (= 1MB 制限想定) | `wc -c` | 棄却: 222 KB |
| H3 | metadata json 不整合 | `python -m json.tool` | 棄却: `json OK` |
| H4 | non-ASCII 不正文字 | `grep -P "[^\x00-\x7F]"` | 棄却: 357 line 日本語 comment、 通常範囲 |
| H5 | dataset_source 不存在 | exp008 が同 sources で通っている | 棄却: source 自体は valid |
| H6 | CLI auth 切れ | `kaggle config view` + minimal push | 棄却: auth 正常、 別 slug で即 push 成功 |
| H7 | CLI version 古い | `kaggle --version` | 棄却: `2.1.2` (= 最新近傍) |
| H8 | GPU session 上限 | minimal は GPU off で通る | 棄却: original slug は GPU on/off 関係なく fail |

### 0.3 採用した root cause hypothesis
**slug `rogii-exp010-fold-reform` および `rogii-exp010-fr` が Kaggle 側 server で ghost state**:
- 過去 push 試行時に server 側で **partial registration** が発生し、 list には現れないが save_kernel API は「既存 update」 path を選び、 内部レコードが見つからず `Notebook not found` を返す
- Kaggle CLI source `kaggle_api_extended.py:4214-4238` の `kaggle.kernels.kernels_api_client.save_kernel(request)` は **slug 指定だと update path** に入る挙動
- `kernels list --search "rogii-exp010"` でも UI でも見えないが、 内部 record は「予約」状態

→ **slug を変えると即時通る** = ghost state 仮説と完全整合。

---

## 1. Step 1 結果: syntax + size 検証

```
$ wc -l kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py
4981 ...exp010_fold_reform.py

$ wc -c kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py
222307 ...exp010_fold_reform.py   # = 222 KB << 1 MB 制限

$ .venv/bin/python -m py_compile kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py
# → exit 0、 syntax OK

$ grep -cP "[^\x00-\x7F]" .../exp010_fold_reform.py
357   # = 日本語 comment 357 行、 通常範囲
```

判定: **問題なし**。

---

## 2. Step 2 結果: metadata validation

```
$ python -m json.tool kaggle_kernels/exp010_fold_reform/kernel-metadata.json > /dev/null
# → exit 0、 json valid

$ diff <(jq -c . exp008_case_d_kalman/kernel-metadata.json) <(jq -c . exp010_fold_reform/kernel-metadata.json)
# → 構造差: id / title / code_file のみ、 schema は完全一致
```

dataset_sources の 4 件は exp008 と同一でそちらは正常 push 済 = source 自体は valid。

判定: **問題なし**。

---

## 3. Step 3 結果: CLI version + auth

```
$ .venv/bin/kaggle --version
Kaggle CLI 2.1.2

$ .venv/bin/kaggle config view
- username: ky7240
- auth_method: ACCESS_TOKEN

$ .venv/bin/kaggle kernels list --user ky7240 --search "exp010"
# → rogii-exp010-fold-reform / rogii-exp010-fr どちらも非表示
# → push probe v1 (= 別 slug) のみ表示
```

CLI 健全 / auth 健全 / 既存 slug が UI / list に見えない。

判定: **問題なし**。

---

## 4. Step 4 結果: 別 slug 最小 push test

```
# /tmp/exp010_minimal/ に minimal.py (= print 1 行) + kernel-metadata.json
$ .venv/bin/kaggle kernels push -p /tmp/exp010_minimal
Kernel version 1 successfully pushed.  Please check progress at
https://www.kaggle.com/code/ky7240/rogii-exp010-push-probe

$ .venv/bin/kaggle kernels status ky7240/rogii-exp010-push-probe
ky7240/rogii-exp010-push-probe has status "KernelWorkerStatus.COMPLETE"
```

判定: **CLI / auth / API path は完全に健全**。 失敗は exp010 特定 slug 固有。

---

## 5. Step 5 結果: bisect 削減 push test

Step 4 で「別 slug なら通る」が確認できたため、 **kernel script の bisect は不要**。 代わりに「slug rename」を採用 (= workaround、 §6)。

→ kernel script に変更を加えず push 成功できるため、 bisect は overkill と判断 (= 「軽さで選ぶな」 GM commit を遵守: 真因は slug ghost であり script 不変が最も rule 耐性ある解)。

---

## 6. workaround 実装 + push 成功状態

### 6.1 採用 workaround
**slug を新規 ID にリネーム**:

```
# kaggle_kernels/exp010_fold_reform/kernel-metadata.json
- "id": "ky7240/rogii-exp010-fold-reform"
- "title": "ROGII exp010 Fold Reform"
+ "id": "ky7240/rogii-exp010-foldreform-v1"
+ "title": "ROGII exp010 foldreform v1"
```

kernel script (= `exp010_fold_reform.py`、 4981 行、 222 KB) は **完全に無変更**。
dataset_sources / GPU / internet 設定も無変更。

### 6.2 push 結果
```
$ .venv/bin/kaggle kernels push -p kaggle_kernels/exp010_fold_reform
Kernel version 1 successfully pushed.  Please check progress at
https://www.kaggle.com/code/ky7240/rogii-exp010-foldreform-v1

$ .venv/bin/kaggle kernels status ky7240/rogii-exp010-foldreform-v1
ky7240/rogii-exp010-foldreform-v1 has status "KernelWorkerStatus.RUNNING"
```

→ **GPU session 取得 + RUNNING 突入確認**。

### 6.3 commit
```
2d47e8d fix(exp010): slug rename → push 成功 (rogii-exp010-foldreform-v1)
```

---

## 7. 残課題 + 5/12 朝 までの action

### 7.1 残課題
1. **kernel COMPLETE 待ち**: RUNNING → COMPLETE まで通常 30-60 min (= exp008 と同程度想定)。 fail 時は log 取得 → debug
2. **submit (`submission.csv`) は中央指示まで保留** (= 制約遵守、 quota 5/day を守る)
3. **次回の exp 起票時の予防策**:
   - 新 slug 命名 convention: `rogii-<exp_id>-<short_name>-v<n>` (= `rogii-exp010-foldreform-v1` 形式)
   - 同 slug の push fail 履歴があれば速やかに `-v2` で逃げる
   - 「Notebook not found」が出たら slug ghost を疑い、 即 rename (= bisect より先)

### 7.2 5/12 朝までの action list (= 中央へ渡す)
- [ ] `kernels status ky7240/rogii-exp010-foldreform-v1` で COMPLETE 確認 (= ~14:50 JST + 1 h = ~15:50 JST 目処)
- [ ] COMPLETE 後、 中央指示があれば `kaggle competitions submit` 実行
- [ ] FAIL 時は `kaggle kernels output ky7240/rogii-exp010-foldreform-v1` で log dump、 別 debug session 起動
- [ ] 5/12 朝の submit ready 状態 = output `submission.csv` が dataset として attach されている状態

### 7.3 ROI 評価 (= GM 11 § §11.1 「優勝本質性」)
- workaround cost: metadata 2 行変更 + commit 1 件 = ~2 min
- 数理本質寄与: 直接ゼロ (= push debug、 LB スコア改善は kernel 側で発生)
- ただし **ブロッカー解除**としては critical: exp010 stratified Edge Q + adv val drop の **σ_fold 42% 削減** (= local smoke 4/4 PASS) を LB で検証する前提条件

---

## 8. 関連 file / commit

- kernel: `kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py` (= 4981 行、 4 改修 inject 済)
- metadata: `kaggle_kernels/exp010_fold_reform/kernel-metadata.json` (= slug rename 後)
- 過去設計: `docs/strategy/exp010-fold-reform-design.md` (= a7d5a49)
- 過去 smoke: `df6e1c6 test(exp010): local smoke 4/4 PASS — σ_fold 0.526 → 0.303 (42% 削減)`
- 本 debug commit: `2d47e8d fix(exp010): slug rename → push 成功`

## 9. Kaggle CLI source 参照

- `kaggle/api/kaggle_api_extended.py:4104-4239` `kernels_push()` 本体
  - L4214-4238: `save_kernel(request)` で `request.slug = slug` 指定だと server 側で update path
  - L4254-4285: server response の `result.error` をそのまま print = `Notebook not found` の出所
- L4147-4152: slug 内 `/` 分割で kernel_slug 取得、 username が `ky7240/` 部分
- L4153-4162: title slug 化 (= `slugify()`) と id 不一致を warn のみ (= error にはならない)
