"""
Autonomous Statistical Arbitrage Orchestrator
Runs 24/7 on Render. Auto-scans pairs, auto-launches monitors, sends Telegram notifications.

Environment Variables:
  TELEGRAM_TOKEN   - Bot token from @BotFather
  CHAT_ID          - Your Telegram chat ID
  CAPITAL_PER_PAIR - USD allocated per pair (default: 1000)
  MAX_PAIRS        - Max simultaneous pairs to monitor (default: 3)
  PORT             - HTTP port for health check (default: 10000, set by Render)
"""
import asyncio
import json
import logging
import os
import sys
import csv
import aiohttp
import websockets
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION (All from environment variables for Render)
# ═══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
CAPITAL_PER_PAIR = float(os.environ.get("CAPITAL_PER_PAIR", "1000"))
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", "3"))
FEE_RATE = 0.0004
FUNDING_RATE = 0.0001  # 0.01% per 8 hours
FUNDING_INTERVAL_HOURS = 8
TIME_STOP_DAY_1 = 3   # Close 50% after 3 days
TIME_STOP_DAY_2 = 5   # Close remaining after 5 days
SCAN_INTERVAL_MINUTES = 10
CSV_FILE = "stat_arb_log_v2.csv"
FAPI_URL = "https://fapi.binance.com"

ENTRY_TRANCHES = [(2.0, 0.30), (2.5, 0.30), (3.0, 0.40)]
EXIT_TRANCHES = [(1.0, 0.40), (0.5, 0.30), (0.0, 0.30)]
STOP_LOSS_Z = 5.0

SECTORS = {
    "Layer 1 Classic": ["ADAUSDT", "XRPUSDT", "XLMUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT"],
    "Smart Contracts": ["SOLUSDT", "AVAXUSDT", "DOTUSDT", "NEARUSDT", "ATOMUSDT", "APTUSDT", "SUIUSDT"],
    "DeFi": ["UNIUSDT", "SUSHIUSDT", "AAVEUSDT", "LDOUSDT", "MKRUSDT", "CRVUSDT"],
    "Meme": ["DOGEUSDT", "1000SHIBUSDT", "1000PEPEUSDT", "1000FLOKIUSDT"],
    "Gaming / Metaverse": ["SANDUSDT", "MANAUSDT", "GALAUSDT", "AXSUSDT", "ENJUSDT"],
    "AI / Data": ["FETUSDT", "RNDRUSDT", "THETAUSDT"],
    "Infrastructure": ["LINKUSDT", "FILUSDT", "ARUSDT", "GRTUSDT"],
}

# ═══════════════════════════════════════════════════════════════
#  TELEGRAM NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════
async def send_telegram(message: str):
    """Send a notification to Telegram."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram send failed: {await resp.text()}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ═══════════════════════════════════════════════════════════════
#  PAIR SCANNER
# ═══════════════════════════════════════════════════════════════
async def fetch_klines(session, symbol, interval='1h', limit=1500, sem=None):
    if sem:
        await sem.acquire()
    try:
        url = f"{FAPI_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        async with session.get(url, params=params, timeout=15) as response:
            if response.status != 200:
                logger.error(f"API Error {response.status} for {symbol}: {await response.text()}")
                return None
            data = await response.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['close'] = df['close'].astype(float)
                df.set_index('timestamp', inplace=True)
                return df[['close']]
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
    finally:
        if sem:
            sem.release()
    return None

def test_pair(df1, df2, sym1, sym2):
    df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=(f'_{sym1}', f'_{sym2}'))
    if len(df) < 500:
        return None
    try:
        asset1, asset2 = df[f'close_{sym1}'], df[f'close_{sym2}']
        X = sm.add_constant(asset2)
        model = sm.OLS(asset1, X).fit()
        hedge_ratio = model.params.iloc[1]
        spread = asset1 - (hedge_ratio * asset2)
        adf_result = adfuller(spread, autolag='AIC')
        p_value = adf_result[1]
        if p_value >= 0.05:
            return None
        window = 480
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_series = (spread - rolling_mean) / rolling_std
        current_z = z_series.iloc[-1]
        return {
            "sym1": sym1, "sym2": sym2,
            "pair_key": f"{sym1}_{sym2}",
            "display": f"{sym1.replace('USDT','')}/{sym2.replace('USDT','')}",
            "p_value": round(p_value, 4),
            "hedge_ratio": hedge_ratio,
            "z_score": round(current_z, 2),
            "abs_z": round(abs(current_z), 2),
            "spread_mean": rolling_mean.iloc[-1],
            "spread_std": rolling_std.iloc[-1],
        }
    except Exception:
        return None

async def scan_all_pairs():
    """Scan all sector pairs and return ranked results."""
    all_symbols = set()
    for syms in SECTORS.values():
        all_symbols.update(syms)
    all_symbols = sorted(all_symbols)

    logger.info(f"📡 Scanner: Downloading {len(all_symbols)} symbols...")
    price_data = {}
    
    # 💥 CHANGE: Limit concurrency to avoid being dropped by Binance
    sem = asyncio.Semaphore(5)
    
    async with aiohttp.ClientSession() as session:
        tasks_dict = {sym: fetch_klines(session, sym, sem=sem) for sym in all_symbols}
        results = await asyncio.gather(*tasks_dict.values())
        for sym, result in zip(tasks_dict.keys(), results):
            if result is not None:
                price_data[sym] = result

    logger.info(f"📡 Scanner: Download complete. Successfully fetched {len(price_data)}/{len(all_symbols)} symbols.")
    logger.info(f"📡 Scanner: Running cointegration math on free-tier CPU... (this may take up to 60s)")

    winners = []
    pairs_tested = 0
    for sector_syms in SECTORS.values():
        available = [s for s in sector_syms if s in price_data]
        for s1, s2 in combinations(available, 2):
            result = test_pair(price_data[s1], price_data[s2], s1, s2)
            if result:
                winners.append(result)
            pairs_tested += 1
            
            # 🔥 CHANGE: Yield event loop every single pair so the Render HTTP server 
            # can answer health checks instantly and not get killed for being unresponsive.
            if pairs_tested % 50 == 0:
                logger.info(f"📡 Scanner progress: {pairs_tested} pairs analyzed...")
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0)

    winners.sort(key=lambda x: x["abs_z"], reverse=True)
    logger.info(f"📡 Scanner: Math complete! Found {len(winners)} cointegrated pairs out of {pairs_tested} tested.")
    return winners

# ═══════════════════════════════════════════════════════════════
#  CSV LOGGING
# ═══════════════════════════════════════════════════════════════
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Exit Date", "Pair", "Action", "Exit Type", "Qty Closed %",
                         "Avg Price 1", "Avg Price 2", "Exit Price 1", "Exit Price 2",
                         "Gross PnL ($)", "Fees ($)", "Funding ($)", "Net PnL ($)",
                         "Reason", "Cumulative Balance ($)"])

# ═══════════════════════════════════════════════════════════════
#  MONITOR INSTANCE (one per pair, runs as an asyncio task)
# ═══════════════════════════════════════════════════════════════
class PairMonitor:
    """Manages a single pair's trading lifecycle."""

    def __init__(self, pair_info: dict):
        self.sym1 = pair_info["sym1"]
        self.sym2 = pair_info["sym2"]
        self.pair_key = pair_info["pair_key"]
        self.display = pair_info["display"]
        self.hedge_ratio = pair_info["hedge_ratio"]
        self.spread_mean = pair_info["spread_mean"]
        self.spread_std = pair_info["spread_std"]
        self.prices = {self.sym1: 0.0, self.sym2: 0.0}
        self.ticks = 0
        self.active = True
        self.balance_contribution = 0.0  # Net PnL contribution to global balance
        self.launched_at = datetime.now()  # Track when monitor was launched
        self.SEARCH_TIMEOUT_MINUTES = 30   # Auto-close if no entry after 30 min

        self.state_file = f"state_{self.pair_key}.json"
        self.position = {
            "side": None, "tranches_filled": 0, "exits_done": 0,
            "spent_1": 0.0, "spent_2": 0.0,
            "qty_1": 0.0, "qty_2": 0.0,
            "original_qty_1": 0.0, "original_qty_2": 0.0,
            "entry_time": None,
            "last_funding_time": None,
            "total_funding_paid": 0.0,
        }
        self._load_state()

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump({"position": self.position, "balance_contribution": self.balance_contribution}, f, indent=2, default=str)

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.position.update(data.get("position", {}))
                    self.balance_contribution = data.get("balance_contribution", 0.0)
                if self.position["tranches_filled"] > 0:
                    logger.info(f"🔄 [{self.display}] Resumed trade: {self.position['side']} | Entries: {self.position['tranches_filled']}/{len(ENTRY_TRANCHES)} | Exits: {self.position['exits_done']}/{len(EXIT_TRANCHES)}")
            except Exception as e:
                logger.error(f"Failed loading state for {self.display}: {e}")

    def _reset_position(self):
        self.position.update({
            "side": None, "tranches_filled": 0, "exits_done": 0,
            "spent_1": 0.0, "spent_2": 0.0,
            "qty_1": 0.0, "qty_2": 0.0,
            "original_qty_1": 0.0, "original_qty_2": 0.0,
            "entry_time": None, "last_funding_time": None, "total_funding_paid": 0.0,
        })

    async def _log_trade(self, p1, p2, gross_pnl, fees, funding, net_pnl, reason, exit_pct):
        self.balance_contribution += net_pnl
        exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_1 = self.position["spent_1"] / self.position["original_qty_1"] if self.position["original_qty_1"] > 0 else 0
        avg_2 = self.position["spent_2"] / self.position["original_qty_2"] if self.position["original_qty_2"] > 0 else 0
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                exit_time, self.display, self.position['side'],
                f"Exit {self.position['exits_done']}/{len(EXIT_TRANCHES)}",
                f"{exit_pct*100:.0f}%",
                round(avg_1, 4), round(avg_2, 4), round(p1, 4), round(p2, 4),
                round(gross_pnl, 2), round(fees, 2), round(funding, 2), round(net_pnl, 2),
                reason, round(CAPITAL_PER_PAIR + self.balance_contribution, 2)
            ])
        logger.info(f"💾 [{self.display}] Logged ({reason}) | PnL: ${net_pnl:.2f}")

    async def _execute_entry(self, z_score, side, idx):
        if self.position["tranches_filled"] == 0:
            self.position["side"] = side
            self.position["entry_time"] = datetime.now().isoformat()
            self.position["last_funding_time"] = datetime.now().isoformat()

        pct = ENTRY_TRANCHES[idx][1]
        capital = CAPITAL_PER_PAIR * pct
        half = capital / 2.0
        p1, p2 = self.prices[self.sym1], self.prices[self.sym2]
        q1, q2 = half / p1, half / p2

        self.position["qty_1"] += q1
        self.position["qty_2"] += q2
        self.position["spent_1"] += half
        self.position["spent_2"] += half
        self.position["tranches_filled"] += 1
        self.position["original_qty_1"] = self.position["qty_1"]
        self.position["original_qty_2"] = self.position["qty_2"]

        total_invested = self.position["spent_1"] + self.position["spent_2"]
        msg = (f"🏦 <b>ENTRY {idx+1}/{len(ENTRY_TRANCHES)}</b> | {self.display}\n"
               f"Side: {side} | Z: {z_score:.2f}\n"
               f"Capital: ${capital:.0f} | Total Invested: ${total_invested:.0f}\n"
               f"Avg {self.sym1}: {p1:.4f}")
        logger.warning(f"🏦 [{self.display}] ENTRY {idx+1}/{len(ENTRY_TRANCHES)} | {side} | Z: {z_score:.2f} | ${capital:.0f}")
        await send_telegram(msg)
        self._save_state()

    async def _execute_partial_exit(self, exit_idx, reason="TAKE_PROFIT"):
        p1, p2 = self.prices[self.sym1], self.prices[self.sym2]
        avg_1 = self.position["spent_1"] / self.position["original_qty_1"]
        avg_2 = self.position["spent_2"] / self.position["original_qty_2"]

        exit_pct = EXIT_TRANCHES[exit_idx][1]
        close_q1 = self.position["original_qty_1"] * exit_pct
        close_q2 = self.position["original_qty_2"] * exit_pct

        if self.position["side"] == "LONG_1_SHORT_2":
            pnl = (p1 - avg_1) * close_q1 + (avg_2 - p2) * close_q2
        else:
            pnl = (avg_1 - p1) * close_q1 + (p2 - avg_2) * close_q2

        volume = (close_q1 * p1 + close_q2 * p2) * 2
        fees = volume * FEE_RATE
        funding_portion = self.position["total_funding_paid"] * exit_pct
        net = pnl - fees - funding_portion

        self.position["exits_done"] += 1
        self.position["qty_1"] -= close_q1
        self.position["qty_2"] -= close_q2

        pair_balance = CAPITAL_PER_PAIR + self.balance_contribution + net
        msg = (f"🟢 <b>EXIT {self.position['exits_done']}/{len(EXIT_TRANCHES)}</b> | {self.display}\n"
               f"Reason: {reason} | Closed: {exit_pct*100:.0f}%\n"
               f"Gross: ${pnl:.2f} | Fees: ${fees:.2f} | Funding: ${funding_portion:.2f}\n"
               f"<b>NET PnL: ${net:.2f}</b>\n"
               f"💼 Pair Balance: ${pair_balance:.2f}")
        logger.info(f"🟢 [{self.display}] EXIT {self.position['exits_done']}/{len(EXIT_TRANCHES)} | {reason} | NET: ${net:.2f}")
        await send_telegram(msg)
        await self._log_trade(p1, p2, pnl, fees, funding_portion, net, reason, exit_pct)

        if self.position["exits_done"] >= len(EXIT_TRANCHES):
            logger.info(f"✅ [{self.display}] Position fully closed.")
            self._reset_position()
            self.active = False  # Signal orchestrator to free the slot

        self._save_state()

    async def _execute_full_exit(self, reason="STOP_LOSS"):
        p1, p2 = self.prices[self.sym1], self.prices[self.sym2]
        avg_1 = self.position["spent_1"] / self.position["original_qty_1"]
        avg_2 = self.position["spent_2"] / self.position["original_qty_2"]
        rq1, rq2 = self.position["qty_1"], self.position["qty_2"]
        remaining_pct = rq1 / self.position["original_qty_1"] if self.position["original_qty_1"] > 0 else 1.0

        if self.position["side"] == "LONG_1_SHORT_2":
            pnl = (p1 - avg_1) * rq1 + (avg_2 - p2) * rq2
        else:
            pnl = (avg_1 - p1) * rq1 + (p2 - avg_2) * rq2

        volume = (rq1 * p1 + rq2 * p2) * 2
        fees = volume * FEE_RATE
        funding = self.position["total_funding_paid"] * remaining_pct
        net = pnl - fees - funding

        pair_balance = CAPITAL_PER_PAIR + self.balance_contribution + net
        msg = (f"🔴 <b>{reason}</b> | {self.display}\n"
               f"Closing remaining {remaining_pct*100:.0f}%\n"
               f"Gross: ${pnl:.2f} | Fees: ${fees:.2f} | Funding: ${funding:.2f}\n"
               f"<b>NET PnL: ${net:.2f}</b>\n"
               f"💼 Pair Balance: ${pair_balance:.2f}")
        logger.info(f"🔴 [{self.display}] {reason} | NET: ${net:.2f}")
        await send_telegram(msg)
        await self._log_trade(p1, p2, pnl, fees, funding, net, reason, remaining_pct)
        self._reset_position()
        self.active = False
        self._save_state()

    def _check_funding_rate(self):
        """Deduct funding rate every 8 hours."""
        if self.position["tranches_filled"] == 0 or not self.position["last_funding_time"]:
            return
        last = datetime.fromisoformat(self.position["last_funding_time"])
        now = datetime.now()
        hours_passed = (now - last).total_seconds() / 3600
        if hours_passed >= FUNDING_INTERVAL_HOURS:
            cycles = int(hours_passed // FUNDING_INTERVAL_HOURS)
            p1, p2 = self.prices[self.sym1], self.prices[self.sym2]
            position_value = self.position["qty_1"] * p1 + self.position["qty_2"] * p2
            charge = position_value * FUNDING_RATE * cycles
            self.position["total_funding_paid"] += charge
            self.position["last_funding_time"] = now.isoformat()
            logger.info(f"💸 [{self.display}] Funding charged: ${charge:.2f} ({cycles}x8h) | Total funding: ${self.position['total_funding_paid']:.2f}")
            self._save_state()

    def _check_time_stop(self):
        """Check if position has been open too long (Option B exits)."""
        if self.position["tranches_filled"] == 0 or not self.position["entry_time"]:
            return None
        entry = datetime.fromisoformat(self.position["entry_time"])
        days_open = (datetime.now() - entry).total_seconds() / 86400

        if days_open >= TIME_STOP_DAY_2:
            return "TIME_STOP_D5"
        elif days_open >= TIME_STOP_DAY_1:
            return "TIME_STOP_D3"
        return None

    async def process_tick(self):
        self.ticks += 1
        if self.prices[self.sym1] == 0.0 or self.prices[self.sym2] == 0.0:
            return

        spread = self.prices[self.sym1] - (self.hedge_ratio * self.prices[self.sym2])
        z_score = (spread - self.spread_mean) / self.spread_std
        abs_z = abs(z_score)
        expected_side = "SHORT_1_LONG_2" if z_score > 0 else "LONG_1_SHORT_2"

        # Funding rate check
        self._check_funding_rate()

        # SEARCH TIMEOUT: If no entry after 30 min, free the slot for better pairs
        if self.position["tranches_filled"] == 0:
            minutes_searching = (datetime.now() - self.launched_at).total_seconds() / 60
            if minutes_searching >= self.SEARCH_TIMEOUT_MINUTES:
                logger.info(f"⏳ [{self.display}] No entry after {self.SEARCH_TIMEOUT_MINUTES}min. Freeing slot.")
                self.active = False
                return

        # Block entry if market broken
        if abs_z >= STOP_LOSS_Z and self.position["tranches_filled"] == 0:
            return

        # STOP LOSS
        if self.position["tranches_filled"] > 0 and abs_z >= STOP_LOSS_Z:
            await self._execute_full_exit("STOP_LOSS")
            return

        # TIME STOP CHECK
        if self.position["tranches_filled"] > 0:
            time_stop = self._check_time_stop()
            if time_stop == "TIME_STOP_D5":
                await self._execute_full_exit("TIME_STOP_D5")
                return
            elif time_stop == "TIME_STOP_D3" and self.position["exits_done"] == 0:
                # Close 50% of position at day 3
                await self._execute_time_partial("TIME_STOP_D3", 0.50)
                return

        # SCALED TAKE PROFIT
        if self.position["tranches_filled"] > 0 and self.position["exits_done"] < len(EXIT_TRANCHES):
            exit_idx = self.position["exits_done"]
            if abs_z <= EXIT_TRANCHES[exit_idx][0]:
                await self._execute_partial_exit(exit_idx, "TAKE_PROFIT")
                return

        # ENTRY EVALUATION
        if self.position["tranches_filled"] < len(ENTRY_TRANCHES) and self.position["exits_done"] == 0:
            next_idx = self.position["tranches_filled"]
            target_z = ENTRY_TRANCHES[next_idx][0]
            if abs_z >= target_z:
                capital = CAPITAL_PER_PAIR * ENTRY_TRANCHES[next_idx][1]
                q1 = (capital / 2.0) / self.prices[self.sym1]
                deviation = abs(spread - self.spread_mean)
                expected_pnl = deviation * q1 * 2
                est_fees = (capital * 2) * FEE_RATE
                if expected_pnl > (est_fees * 1.5):
                    if self.position["tranches_filled"] == 0 or self.position["side"] == expected_side:
                        await self._execute_entry(z_score, expected_side, next_idx)

        # Periodic log
        if self.ticks % 500 == 0:
            if self.position["tranches_filled"] > 0:
                state = f"In:{self.position['tranches_filled']}/{len(ENTRY_TRANCHES)} Out:{self.position['exits_done']}/{len(EXIT_TRANCHES)}"
            else:
                state = "Searching"
            logger.info(f"📡 [{self.display}] Z: {z_score:.2f} | {state}")

    async def _execute_time_partial(self, reason, pct):
        """Close a percentage of remaining position due to time stop."""
        p1, p2 = self.prices[self.sym1], self.prices[self.sym2]
        avg_1 = self.position["spent_1"] / self.position["original_qty_1"]
        avg_2 = self.position["spent_2"] / self.position["original_qty_2"]
        close_q1 = self.position["qty_1"] * pct
        close_q2 = self.position["qty_2"] * pct

        if self.position["side"] == "LONG_1_SHORT_2":
            pnl = (p1 - avg_1) * close_q1 + (avg_2 - p2) * close_q2
        else:
            pnl = (avg_1 - p1) * close_q1 + (p2 - avg_2) * close_q2

        volume = (close_q1 * p1 + close_q2 * p2) * 2
        fees = volume * FEE_RATE
        funding = self.position["total_funding_paid"] * pct
        net = pnl - fees - funding

        self.position["qty_1"] -= close_q1
        self.position["qty_2"] -= close_q2
        self.position["exits_done"] += 1

        msg = f"⏰ <b>{reason}</b> | {self.display}\nClosing {pct*100:.0f}% (time-based)\nGross: ${pnl:.2f} | Fees: ${fees:.2f} | Funding: ${funding:.2f}\n<b>NET PnL: ${net:.2f}</b>"
        logger.info(f"⏰ [{self.display}] {reason} | Closing {pct*100:.0f}% | NET: ${net:.2f}")
        await send_telegram(msg)
        await self._log_trade(p1, p2, pnl, fees, funding, net, reason, pct)
        self._save_state()

    async def run(self):
        """Main WebSocket loop for this pair."""
        s1, s2 = self.sym1.lower(), self.sym2.lower()
        url = f"wss://fstream.binance.com/ws/{s1}@bookTicker/{s2}@bookTicker"
        while self.active:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    logger.info(f"🔗 [{self.display}] WebSocket connected")
                    while self.active:
                        data = json.loads(await ws.recv())
                        symbol = data.get("s")
                        mid = (float(data["b"]) + float(data["a"])) / 2.0
                        if symbol == self.sym1:
                            self.prices[self.sym1] = mid
                        elif symbol == self.sym2:
                            self.prices[self.sym2] = mid
                        await self.process_tick()
            except Exception as e:
                if self.active:
                    logger.error(f"[{self.display}] WS Error: {e}. Reconnecting...")
                    await asyncio.sleep(3)

# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATOR (Main loop)
# ═══════════════════════════════════════════════════════════════
class Orchestrator:
    def __init__(self):
        self.active_monitors: dict[str, asyncio.Task] = {}
        self.monitor_instances: dict[str, PairMonitor] = {}

    def active_count(self):
        # Clean up finished tasks
        finished = [k for k, t in self.active_monitors.items() if t.done()]
        for k in finished:
            del self.active_monitors[k]
            if k in self.monitor_instances:
                del self.monitor_instances[k]
        return len(self.active_monitors)

    async def launch_monitor(self, pair_info):
        key = pair_info["pair_key"]
        if key in self.active_monitors and not self.active_monitors[key].done():
            return  # Already running

        monitor = PairMonitor(pair_info)
        self.monitor_instances[key] = monitor
        task = asyncio.create_task(monitor.run())
        self.active_monitors[key] = task
        logger.info(f"🚀 Launched monitor for {pair_info['display']} (Z: {pair_info['z_score']})")

    async def scan_and_launch(self):
        """Scan for pairs and auto-launch monitors for hot opportunities."""
        try:
            winners = await scan_all_pairs()
            hot_pairs = [w for w in winners if w["abs_z"] >= 2.0]
            
            msg = f"📡 <b>Scanner Update</b>\nFound {len(winners)} cointegrated pairs."
            if hot_pairs:
                names = ", ".join([w["display"] for w in hot_pairs[:5]])
                msg += f"\n🔥 Found {len(hot_pairs)} hot pairs (|Z| ≥ 2.0):\n{names}"
            else:
                msg += f"\n🧊 No pairs found with |Z| ≥ 2.0 right now."
                
            await send_telegram(msg)

            slots = MAX_PAIRS - self.active_count()
            launched = 0
            for pair in hot_pairs:
                if slots <= 0:
                    break
                if pair["pair_key"] not in self.active_monitors:
                    await self.launch_monitor(pair)
                    slots -= 1
                    launched += 1

            if launched > 0:
                logger.info(f"🚀 Launched {launched} new monitors. Active: {self.active_count()}/{MAX_PAIRS}")
        except Exception as e:
            logger.error(f"Error in scan_and_launch: {e}")
            await send_telegram(f"⚠️ <b>Scanner Error</b>\n{e}")

    async def run(self):
        logger.info("=" * 60)
        logger.info("  AUTONOMOUS STAT ARB ORCHESTRATOR")
        logger.info(f"  Capital/Pair: ${CAPITAL_PER_PAIR} | Max Pairs: {MAX_PAIRS}")
        logger.info(f"  Telegram: {'✅ Configured' if TELEGRAM_TOKEN else '❌ Not set'}")
        logger.info("=" * 60)

        await send_telegram(
            f"🤖 <b>Bot Started</b>\n"
            f"Capital/Pair: ${CAPITAL_PER_PAIR}\n"
            f"Max Pairs: {MAX_PAIRS}\n"
            f"Scan Interval: {SCAN_INTERVAL_MINUTES}min"
        )

        # Initial scan
        await self.scan_and_launch()

        # Launch daily summary and scanner as concurrent tasks
        asyncio.create_task(self._daily_summary_loop())

        # Periodic scanning loop
        while True:
            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)
            await self.scan_and_launch()

    async def _daily_summary_loop(self):
        """Send a daily summary report to Telegram every 24 hours."""
        while True:
            await asyncio.sleep(86400)  # 24 hours
            try:
                await self._send_summary()
            except Exception as e:
                logger.error(f"Daily summary error: {e}")

    async def _send_summary(self):
        """Build and send a portfolio summary to Telegram."""
        lines = ["📊 <b>DAILY PORTFOLIO SUMMARY</b>\n"]
        total_balance = 0.0
        active_pairs = 0

        for key, monitor in self.monitor_instances.items():
            pair_bal = CAPITAL_PER_PAIR + monitor.balance_contribution
            total_balance += pair_bal
            pnl_str = f"+${monitor.balance_contribution:.2f}" if monitor.balance_contribution >= 0 else f"-${abs(monitor.balance_contribution):.2f}"

            if monitor.position["tranches_filled"] > 0:
                active_pairs += 1
                spread = monitor.prices[monitor.sym1] - (monitor.hedge_ratio * monitor.prices[monitor.sym2])
                z = (spread - monitor.spread_mean) / monitor.spread_std if monitor.spread_std != 0 else 0
                funding = monitor.position['total_funding_paid']
                lines.append(f"  📍 <b>{monitor.display}</b>: Z={z:.2f} | In:{monitor.position['tranches_filled']}/{len(ENTRY_TRANCHES)} Out:{monitor.position['exits_done']}/{len(EXIT_TRANCHES)} | Funding: ${funding:.2f}")
            else:
                lines.append(f"  🔍 <b>{monitor.display}</b>: Searching")

            lines.append(f"     PnL: {pnl_str} | Balance: ${pair_bal:.2f}")

        idle_capital = (MAX_PAIRS - len(self.monitor_instances)) * CAPITAL_PER_PAIR
        total_balance += idle_capital

        lines.append(f"\n💰 <b>Total Balance: ${total_balance:.2f}</b>")
        lines.append(f"📈 Active Pairs: {active_pairs}/{MAX_PAIRS}")
        lines.append(f"💤 Idle Capital: ${idle_capital:.2f}")

        await send_telegram("\n".join(lines))

# ═══════════════════════════════════════════════════════════════
#  HTTP HEALTH SERVER (keeps Render free tier alive via UptimeRobot)
# ═══════════════════════════════════════════════════════════════
async def health_handler(request):
    """Returns bot status as JSON. UptimeRobot pings this every 5 min."""
    orch = request.app.get("orchestrator")
    status = {
        "status": "running",
        "active_pairs": orch.active_count() if orch else 0,
        "max_pairs": MAX_PAIRS,
        "capital_per_pair": CAPITAL_PER_PAIR,
    }
    if orch:
        status["monitors"] = []
        for key, monitor in orch.monitor_instances.items():
            m = {
                "pair": monitor.display,
                "in_trade": monitor.position["tranches_filled"] > 0,
                "entries": f"{monitor.position['tranches_filled']}/{len(ENTRY_TRANCHES)}",
                "exits": f"{monitor.position['exits_done']}/{len(EXIT_TRANCHES)}",
                "pnl": round(monitor.balance_contribution, 2),
            }
            status["monitors"].append(m)
    from aiohttp import web
    return web.json_response(status)

async def start_http_server(orchestrator_instance):
    from aiohttp import web
    app = web.Application()
    app["orchestrator"] = orchestrator_instance
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    port = int(os.environ.get("PORT", "10000"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"🌐 Health server running on port {port}")

async def main():
    orchestrator = Orchestrator()
    await start_http_server(orchestrator)
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())
