import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import concurrent.futures
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm

# ==================================================================================
# CONFIGURAZIONE FINANCE PRO DASHBOARD ⚡ (INSTITUTIONAL GRADE)
# ==================================================================================
FTMO_ACCOUNT_ID = 1521085130
FTMO_PASSWORD = "$2p*I!V$2Dpp?"
FTMO_SERVER = "FTMO-Demo2"

PAIRS = [
    ("EURUSD", "GBPUSD", 1.0),
    ("XAUUSD", "XAGUSD", 0.5),
    ("US500.cash", "US100.cash", 0.1),
    ("BTCUSD", "ETHUSD", 0.01)
]

# --- Parametri Quantitativi ---
# [FIX 1] Unità temporale esplicita: Candele (Minuti su M1)
MAX_HALF_LIFE_CANDLES = 20  # Max 20 minuti per il mean reversion atteso
MIN_CONFIDENCE = 0.60
Z_ENTRY = 2.0
Z_EXIT = 0.5
MIN_LOOKBACK = 100
MAX_LOOKBACK = 300

# --- Risk Management FTMO ---
DAILY_LOSS_LIMIT_PCT = 0.04
TOTAL_LOSS_LIMIT_PCT = 0.09
VAR_CONFIDENCE = 1.96       # 95% Confidence Interval
MAX_SWAP_IMPACT = 0.2

# [FIX 3] Blacklist per coppie con rottura strutturale (Structural Break)
BLACKLIST = set()

# ==================================================================================
# MODULO ORNSTEIN-UHLENBECK
# ==================================================================================


class OrnsteinUhlenbeck:
    @staticmethod
    def fit(spread_series, dt=1.0):
        # dt = 1.0 implica che theta è in unità "per barra" (per minuto su M1)
        x_t = spread_series[:-1].values
        dx_t = spread_series[1:].values - x_t

        A = np.vstack([x_t, np.ones(len(x_t))]).T
        beta, alpha = np.linalg.lstsq(A, dx_t, rcond=None)[0]

        theta = -beta / dt
        mu = alpha / (theta * dt) if abs(theta) > 1e-5 else 0

        residuals = dx_t - (alpha + beta * x_t)
        sigma = np.std(residuals) / np.sqrt(dt)

        return theta, mu, sigma

    @staticmethod
    def calculate_metrics(theta, mu, sigma, current_spread):
        if theta <= 1e-5:
            return 999, 0.0

        hl = np.log(2) / theta

        sigma_eq = sigma / np.sqrt(2 * theta)
        distance = abs(current_spread - mu)
        p_conv = 2 * (1 - norm.cdf(distance / sigma_eq))

        return hl, p_conv

# ==================================================================================
# CLASSE KALMAN FILTER
# ==================================================================================


class KalmanFilterReg:
    def __init__(self):
        self.delta = 1e-4
        self.wt = self.delta / (1 - self.delta) * np.eye(2)
        self.vt = 1e-3
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = None

    def update(self, x, y):
        F = np.asarray([x, 1.0]).reshape(1, 2)
        y = np.asarray(y)

        if self.R is None:
            self.R = np.zeros((2, 2))
            self.theta = np.zeros(2)

        y_hat = F @ self.theta
        e = y - y_hat
        Q = self.P + self.wt
        R_val = self.vt

        K = Q @ F.T / (F @ Q @ F.T + R_val)
        self.theta = self.theta + (K.flatten() * e)
        self.P = (np.eye(2) - K @ F) @ Q

        return self.theta[0], self.theta[1], e

# ==================================================================================
# RISK MANAGER
# ==================================================================================


class RiskManager:
    @staticmethod
    def check_drawdown_limits(initial_balance, current_equity):
        drawdown = (initial_balance - current_equity) / initial_balance
        if drawdown >= TOTAL_LOSS_LIMIT_PCT:
            print(
                f"🚨 CRITICAL: Max Drawdown raggiunto ({drawdown:.2%}). STOP TRADING.")
            return False
        return True

    @staticmethod
    def calculate_convex_size(equity, initial_balance, sigma_ou, point_value_a):
        # [FIX 2] Nota sulla Volatilità:
        # sigma_ou è calcolata sugli ultimi 300 min (Regime Intraday Corrente).
        # Scalare per sqrt(1440) proietta questo regime su base giornaliera.
        # È una stima conservativa se il mercato è in un regime di alta volatilità.

        limit_equity = initial_balance * (1 - TOTAL_LOSS_LIMIT_PCT)
        distance_to_ruin = max(0, equity - limit_equity)

        if distance_to_ruin <= 0:
            return 0.0

        vol_daily = sigma_ou * np.sqrt(1440)
        var_price_impact = vol_daily * VAR_CONFIDENCE
        var_money_per_lot = var_price_impact * point_value_a

        if var_money_per_lot == 0:
            return 0.0

        kelly_fraction = 0.1
        raw_risk_capital = distance_to_ruin * kelly_fraction
        lot_size = raw_risk_capital / var_money_per_lot

        lot_size = min(10.0, lot_size)
        return max(0.01, np.floor(lot_size * 100) / 100)

    @staticmethod
    def check_swap_costs(symbol_a, symbol_b, side_a):
        s_a = mt5.symbol_info(symbol_a)
        s_b = mt5.symbol_info(symbol_b)
        if not s_a or not s_b:
            return False

        swap_a = s_a.swap_long if side_a == 'buy' else s_a.swap_short
        swap_b = s_b.swap_short if side_a == 'buy' else s_b.swap_long

        total_swap = swap_a + swap_b
        if total_swap < -5.0:
            # print(f"⚠️ SWAP SKIP {symbol_a}/{symbol_b}: {total_swap}")
            return False
        return True

# ==================================================================================
# CORE LOGIC
# ==================================================================================


def process_pair(pair_config, account_currency, initial_balance):
    asset_a, asset_b, initial_hedge = pair_config
    pair_id = f"{asset_a}/{asset_b}"

    # [FIX 3] Check Blacklist
    if pair_id in BLACKLIST:
        return

    # Data Check
    if not mt5.symbol_info(asset_a).visible:
        mt5.symbol_select(asset_a, True)
    if not mt5.symbol_info(asset_b).visible:
        mt5.symbol_select(asset_b, True)

    rates_a = mt5.copy_rates_from_pos(
        asset_a, mt5.TIMEFRAME_M1, 0, MAX_LOOKBACK)
    rates_b = mt5.copy_rates_from_pos(
        asset_b, mt5.TIMEFRAME_M1, 0, MAX_LOOKBACK)

    if rates_a is None or rates_b is None or len(rates_a) != len(rates_b):
        return

    df = pd.DataFrame({'A': [x['close'] for x in rates_a], 'B': [
                      x['close'] for x in rates_b]})

    # Kalman
    kf = KalmanFilterReg()
    spreads = []
    betas = []

    for i in range(len(df)):
        b, alpha, e = kf.update(df['B'].iloc[i], df['A'].iloc[i])
        s = df['A'].iloc[i] - (b * df['B'].iloc[i])
        spreads.append(s)
        betas.append(b)

    current_spread = spreads[-1]
    current_beta = betas[-1]
    series_spread = pd.Series(spreads[-100:])

    # Diagnostica
    try:
        adf = adfuller(series_spread)
        adf_p = adf[1]

        theta, mu, sigma_ou = OrnsteinUhlenbeck.fit(series_spread)
        half_life, p_conv = OrnsteinUhlenbeck.calculate_metrics(
            theta, mu, sigma_ou, current_spread)
    except:
        return

    # [FIX 1] Filtro Half-Life in Candele
    if adf_p > 0.05:
        return
    if half_life > MAX_HALF_LIFE_CANDLES:
        return

    # Z-Score
    mean = series_spread.mean()
    std = series_spread.std()
    if std == 0:
        return
    z_score = (current_spread - mean) / std

    # Execution Logic
    positions = mt5.positions_get(symbol=asset_a)
    has_pos = len(positions) > 0

    if not has_pos:
        if abs(z_score) > Z_ENTRY:
            point_value = mt5.symbol_info(asset_a).point
            acc_info = mt5.account_info()
            lot_size = RiskManager.calculate_convex_size(
                acc_info.equity, initial_balance, std, point_value)

            if lot_size >= 0.01:
                if z_score > Z_ENTRY and RiskManager.check_swap_costs(asset_a, asset_b, 'sell'):
                    print(
                        f"📉 OPEN SHORT {asset_a}/{asset_b} | Z:{z_score:.2f} | HL:{half_life:.1f}m | P:{p_conv:.0%}")
                    execute_atomic_trade(
                        asset_a, asset_b, mt5.ORDER_TYPE_SELL, lot_size, current_beta)
                elif z_score < -Z_ENTRY and RiskManager.check_swap_costs(asset_a, asset_b, 'buy'):
                    print(
                        f"📈 OPEN LONG {asset_a}/{asset_b} | Z:{z_score:.2f} | HL:{half_life:.1f}m | P:{p_conv:.0%}")
                    execute_atomic_trade(
                        asset_a, asset_b, mt5.ORDER_TYPE_BUY, lot_size, current_beta)

    else:
        # Gestione Posizione Aperta
        pos_type = positions[0].type

        # [FIX 3] HARD STOP LOSS & BLACKLISTING
        # Se Z esplode, qualcosa si è rotto strutturalmente.
        if abs(z_score) > 4.5:
            print(
                f"💀 STRUCTURAL BREAK DETECTED {asset_a}/{asset_b} (Z={z_score:.2f})")
            print(f"🚫 BLACKLISTING PAIR {pair_id} for this session.")
            BLACKLIST.add(pair_id)
            close_all_positions(asset_a, asset_b)
            return

        # Take Profit standard (Isteresi)
        if (pos_type == mt5.ORDER_TYPE_SELL and z_score < Z_EXIT) or \
           (pos_type == mt5.ORDER_TYPE_BUY and z_score > -Z_EXIT):
            print(f"💰 TAKE PROFIT {asset_a} | Z: {z_score:.2f}")
            close_all_positions(asset_a, asset_b)

    # Monitor
    if abs(z_score) > 1.5 and not has_pos:
        print(f"👀 WATCH {asset_a} | Z:{z_score:.1f} | HL:{half_life:.1f}m | P:{p_conv:.0%} | Size:{lot_size if 'lot_size' in locals() else 'N/A'}")


def execute_atomic_trade(sym_a, sym_b, type_a, vol_a, beta):
    vol_b = vol_a * abs(beta)
    vol_b = max(0.01, round(vol_b, 2))
    type_b = mt5.ORDER_TYPE_SELL if type_a == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    req_a = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym_a,
        "volume": float(vol_a),
        "type": type_a,
        "price": mt5.symbol_info_tick(sym_a).ask if type_a == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(sym_a).bid,
        "deviation": 10,
        "type_filling": mt5.ORDER_FILLING_FOK,
        "comment": "MDL Quant Inst",
    }

    res_a = mt5.order_send(req_a)
    if res_a.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Fail Leg A ({sym_a}): {res_a.comment}")
        return

    order_b_filled = False
    for i in range(5):
        req_b = req_a.copy()
        req_b["symbol"] = sym_b
        req_b["volume"] = float(vol_b)
        req_b["type"] = type_b
        req_b["price"] = mt5.symbol_info_tick(
            sym_b).ask if type_b == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(sym_b).bid

        res_b = mt5.order_send(req_b)
        if res_b.retcode == mt5.TRADE_RETCODE_DONE:
            order_b_filled = True
            break
        time.sleep(0.05)

    if not order_b_filled:
        print(f"💀 CRITICAL LEGGING FAILURE: Closing {sym_a} NOW.")
        # Chiusura forzata immediata
        positions = mt5.positions_get(symbol=sym_a)
        for p in positions:
            req_c = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": sym_a, "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": p.ticket, "price": mt5.symbol_info_tick(sym_a).bid if p.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(sym_a).ask
            }
            mt5.order_send(req_c)


def close_all_positions(sym_a, sym_b):
    for sym in [sym_a, sym_b]:
        positions = mt5.positions_get(symbol=sym)
        if positions:
            for pos in positions:
                req = {
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket, "price": mt5.symbol_info_tick(sym).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(sym).ask
                }
                mt5.order_send(req)


def main():
    if not mt5.initialize(login=FTMO_ACCOUNT_ID, password=FTMO_PASSWORD, server=FTMO_SERVER):
        return
    acc = mt5.account_info()
    initial_balance = acc.balance
    print(f"✅ QUANT ENGINE (INSTITUTIONAL) READY. Bal: {initial_balance}")
    print("------------------------------------------------------------------")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(PAIRS))

    try:
        while True:
            current_equity = mt5.account_info().equity
            # Heartbeat informativo
            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{now}] 💓 Eq: {current_equity:.2f} | Actives: {len(PAIRS) - len(BLACKLIST)}/{len(PAIRS)}")

            if not RiskManager.check_drawdown_limits(initial_balance, current_equity):
                break

            futures = []
            for pair in PAIRS:
                futures.append(executor.submit(
                    process_pair, pair, acc.currency, initial_balance))
            concurrent.futures.wait(futures)
            time.sleep(60)
    except KeyboardInterrupt:
        mt5.shutdown()


if __name__ == "__main__":
    main()
