"""거래 기록 — 청산된 매매(매수→매도/손절)를 trades_log.json에 누적. 엑셀 보고용."""
import os, json
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_log.json")

def _load():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("trades", [])
    except Exception:
        return []

def _save(trades):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({"trades": trades}, f, ensure_ascii=False, indent=2)

def log_closed(strategy, code, name, buy_date, buy, sell, profit_pct, reason):
    """청산 거래 1건 기록 (중복 방지: 같은 전략·코드·매수일·매도일이면 스킵)."""
    trades = _load()
    sell_date = datetime.now().strftime("%Y-%m-%d")
    key = (strategy, code, buy_date, sell_date)
    for t in trades:
        if (t.get("strategy"), t.get("code"), t.get("buyDate"), t.get("sellDate")) == key:
            return False
    trades.append({
        "strategy": strategy, "code": code, "name": name,
        "buyDate": buy_date, "sellDate": sell_date,
        "buy": int(buy), "sell": int(sell),
        "profitPct": round(float(profit_pct), 2), "reason": reason,
    })
    _save(trades)
    return True

def load_closed():
    return _load()
