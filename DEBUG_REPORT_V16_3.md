# Alpha Radar V16.3 Architecture Freeze — Debug / Test Report

## 1. 本版目標

這一版不再因單一盤中案例新增零散規則，而是完成一次架構收斂：

- 標的選擇、盤中強度、進場時機、市場風險預算分層。
- 漲停／大漲只影響進場，不重複壓低標的排名。
- Bull / Neutral / Bear 只影響部位上限，不再對 ranking score 加減分。
- 高階策略權重及門檻集中至 `strategy_config.py`。
- 保留完整 1000 檔候選設定與資料蒐集，不用少抓資料換速度。

版本：`v16.3.0-architecture-freeze`

## 2. 主要修改

### A. 避免「追高」被重複處罰

V16.2 的盤中 `intraday_score` 會對單日大漲／遠離盤中均價加入 overheat penalty，`entry_timing` 又再因 +6%、+8%、近漲停、ATR 延伸做一次懲罰。V16.3 已拆開：

- `intraday_engine.py`：只衡量盤中強度與可交易性。
- `policy_engine.py`：集中處理追價、隔夜風險與進場分數。

因此一檔漲停股仍可維持高標的排名，但進場分數會低、部位上限可降為 0%。

### B. 市場 regime 不再改個股排名

已移除 `radar_service.py` 原本：

- BULL 對短／中線 ranking score +2。
- BEAR 對短線 ranking score -4。

現在 regime 只透過 `position_guidance()` 調整風險預算。

### C. 盤中權重依時間成熟

開盤 9:00 附近的即時訊號樣本不足，不應與 12:30 同權重。V16.3 將盤中 overlay 權重依交易時間逐步提高：

- 短線：22% → 28% → 35% → 41% → 45%。
- 中線：6% → 8% → 10% → 13% → 15%。
- 長線：2% → 2.5% → 3% → 4% → 5%。

完整日線模型始終是基底。

### D. 部位上限

UI 新增「部位上限」，其意義是相對於使用者原本規劃的正常部位，不是總資產百分比。部位由：

1. 進場分數；
2. 追價風險；
3. 市場 regime；

共同決定，但不回寫選股分數。

### E. 更新速度

保留 V16.2 的效能優化：

- 每檔技術特徵只建一次，短中長三週期共用。
- SQLite WAL / 共用讀取／批次寫入。
- Streamlit 短 TTL 共用同一批盤中即時報價。
- 盤中不重跑多年模型。

7 次離線基準測試中：

- 共用特徵：median 0.2647 秒。
- 三週期分開重算：median 0.3824 秒。
- median speedup：**1.44x**。

此為純模型計算階段；實際完整更新仍會受 Yahoo、TWSE/TPEx、FinMind 網路回應時間影響。

## 3. 測試結果

最終離線回歸測試：

```text
PASS test_config_weights
PASS test_entry_separation
PASS test_regime_only_changes_position
PASS test_time_matured_intraday_weight
PASS test_limit_up_can_rank_high_but_not_be_buyable
PASS test_offline_single_stock_pipeline
PASS test_utf8_ui_strings_are_clean
PASS test_offline_run_scan_regime_does_not_rank
PASS test_shared_feature_engine_matches_legacy_calls
ALL TESTS PASSED | v16.3.0-architecture-freeze
PASS app_smoke_import_without_streamlit_runtime
```

另執行：

- `python -m py_compile *.py`：通過。
- 核心模組實際 import：通過。
- `app.py` 使用 Streamlit stub 做 import smoke test：通過。
- UTF-8 檢查：`app.py`、`policy_engine.py`、`intraday_engine.py`、`strategy_config.py` 均無 replacement character，且「標的分數／進場分數／追價風險／部位上限」字串存在。

目前執行環境未預裝 Streamlit，且容器沒有外網可即時 pip 安裝，因此未在本容器啟動真正的 Streamlit server；部署所需依賴已完整列入 `requirements.txt`。核心計算、整合掃描、盤中重排與 UI 模組 import 均已離線測試。

## 4. 防止後續架構漂移的規則

後續一週與 Gemini 做即時比較時，原則上不直接修改正式模型。建議保留每天固定時間的 Top 5、標的分數、進場分數、部位上限與隔日／3日／5日結果，等樣本累積後一次檢討。

若之後只需調門檻或權重，優先只改 `strategy_config.py`；不要再把同一個風險條件散落到 `radar_service.py`、`intraday_engine.py` 與 `app.py`。
