
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.stats import chi2
import time
from datetime import datetime

# ==================================================================================
# CONFIGURAZIONE UTENTE (DA COMPILARE)
# ==================================================================================

# 1. CREDENZIALI BROKER
FTMO_ACCOUNT_ID = 1521076547      # <-- INSERISCI IL TUO LOGIN
FTMO_PASSWORD = "J39NF**Ud"  # <-- INSERISCI LA TUA PASSWORD
FTMO_SERVER = "FTMO-Demo2"  # <-- INSERISCI IL SERVER ESATTO

# 2. DIMENSIONE CONTO (FONDAMENTALE PER I CALCOLI)
# Inserisci la taglia nominale del conto acquistato (es. 100000, 200000, 10000)
ACCOUNT_SIZE = 10000.0

# 3. SELETTORE FASE (MODIFICARE IN BASE ALLO STATO ATTUALE)
# 1 = FASE 1 (Target 10% | Max Profit Singolo Trade attivo | Rischio Standard)
# 2 = FASE 2 (Target 5%  | Max Profit Singolo Trade attivo | Rischio Dimezzato)
# 3 = LIVE   (No Target  | No Limiti Profitto | Rischio Standard)
TRADING_MODE = 1

# 4. PORTAFOGLIO COPPIE & VOLATILITÀ
# Format: ("Asset A", "Asset B", Moltiplicatore_Rischio)
# Il Moltiplicatore serve a normalizzare la volatilità (es. Indici muovono più soldi del Forex)
PAIRS = [
    ("EURUSD", "GBPUSD", 1.0),   # Forex Major: 1.0
    ("XAUUSD", "XAGUSD", 0.5),   # Gold/Silver: 0.5 (Più volatili)
    # Indici USA: 0.1 (Contratti pesanti)
    ("US500.cash", "US100.cash", 0.1),
    ("BTCUSD", "ETHUSD", 0.01)
]

# ==================================================================================
# PARAMETRI SISTEMA (NON TOCCARE SE NON SEI UN QUANT)
# ==================================================================================

# Limiti di Rischio (Bufferizzati per sicurezza)
MAX_DAILY_LOSS_PCT = 0.04       # 4.0% (Buffer per il 5% ufficiale)
MAX_TOTAL_LOSS_PCT = 0.09       # 9.0% (Buffer per il 10% ufficiale)

# Parametri Strategia (Dal PDF)
MAX_HALF_LIFE = 24              # Max barre per convergenza (Capital Velocity)
Z_ENTRY = 2.0                   # Soglia Ingresso
Z_EXIT = 0.5                    # Soglia Uscita (Isteresi)
TIMEFRAME = mt5.TIMEFRAME_M15   # Timeframe Operativo
LOOKBACK_WINDOW = 60            # Finestra dati per statistiche

# Rischio Base
BASE_RISK_PER_TRADE = 0.005     # 0.5% del capitale corrente a trade

# ==================================================================================
# CALCOLO AUTOMATICO OBIETTIVI E VINCOLI
# ==================================================================================

# Target Profitto
TARGET_PROFIT_AMOUNT = float('inf')
if TRADING_MODE == 1:
    TARGET_PROFIT_AMOUNT = ACCOUNT_SIZE * 0.10  # 10%
elif TRADING_MODE == 2:
    TARGET_PROFIT_AMOUNT = ACCOUNT_SIZE * 0.05  # 5%

# Limiti Perdita Monetaria
MAX_DAILY_LOSS_AMOUNT = ACCOUNT_SIZE * MAX_DAILY_LOSS_PCT
MAX_TOTAL_LOSS_AMOUNT = ACCOUNT_SIZE * MAX_TOTAL_LOSS_PCT

# Consistency Rule (Max 49% del profitto target in un solo trade)
MAX_SINGLE_TRADE_PROFIT = float('inf')
if TRADING_MODE in [1, 2]:
    MAX_SINGLE_TRADE_PROFIT = TARGET_PROFIT_AMOUNT * 0.49

# ==================================================================================
# MOTORE MATEMATICO (CORE)
# ==================================================================================


class OnlineKalmanFilter:
    """Implementazione ricorsiva del filtro di Kalman [cite: 10, 11]"""

    def __init__(self, delta=1e-4, R=1e-3):
        self.delta = delta
        self.R = R
        self.state = np.zeros(2)  # [beta, intercept]
        self.P = np.eye(2)

    def update(self, y, x):
        # x: Asset B (Indipendente), y: Asset A (Dipendente)
        H = np.array([x, 1.0])
        # Predizione
        self.P = self.P + (np.eye(2) * self.delta)
        # Update
        y_pred = np.dot(H, self.state)
        error = y - y_pred
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(self.P, H.T) / S
        self.state = self.state + K * error
        self.P = np.dot(np.eye(2) - np.outer(K, H), self.P)
        return error, np.sqrt(S)


def calculate_half_life(residuals):
    """Calcola velocità di mean-reversion (Ornstein-Uhlenbeck) [cite: 14]"""
    if len(residuals) < 10:
        return 999
    y_lag = residuals[1:]
    x_lag = residuals[:-1]
    # Regressione lineare semplice
    slope = np.polyfit(x_lag, y_lag, 1)[0]
    if slope >= 1.0 or slope <= 0:
        return 999
    return -np.log(2) / np.log(slope)


def check_ljung_box(residuals, lags=10, alpha=0.05):
    """Test integrità del segnale (Bianchezza dei residui) """
    if len(residuals) < lags + 5:
        return True
    n = len(residuals)
    mean = np.mean(residuals)
    var = np.var(residuals)
    if var == 0:
        return False

    sum_autocorr = 0
    for k in range(1, lags + 1):
        num = np.sum((residuals[k:] - mean) * (residuals[:-k] - mean))
        denom = np.sum((residuals - mean)**2)
        rho_k = num / denom
        sum_autocorr += (rho_k**2) / (n - k)

    q_stat = n * (n + 2) * sum_autocorr
    critical_val = chi2.ppf(1 - alpha, lags)
    return q_stat < critical_val

# ==================================================================================
# GESTIONE RISCHIO AVANZATA (SIZING) [cite: 21]
# ==================================================================================


def get_smart_lot_size(symbol, vol_mult):
    account = mt5.account_info()
    if account is None:
        return 0.0

    equity = account.equity
    current_profit = equity - ACCOUNT_SIZE

    # 1. MODE SCALER (Fase 2 = Rischio dimezzato)
    mode_scaler = 0.5 if TRADING_MODE == 2 else 1.0

    # 2. DISTANCE TO RUIN (Difesa del Capitale)
    # Calcolo quanto spazio ho prima di violare i limiti
    # Daily: (Approssimato sull'equity corrente, in produzione servirebbe reset mezzanotte)
    daily_space = MAX_DAILY_LOSS_AMOUNT - \
        (ACCOUNT_SIZE - equity if equity < ACCOUNT_SIZE else 0)
    # Total:
    total_space = (ACCOUNT_SIZE - MAX_TOTAL_LOSS_AMOUNT) - \
        (ACCOUNT_SIZE - equity)

    # Prendo il limite più vicino
    dist_ruin = min(daily_space, MAX_TOTAL_LOSS_AMOUNT +
                    current_profit)  # Semplificato

    if dist_ruin <= 0:
        return 0.0  # Game Over

    # Safety Factor: Se il buffer si riduce, riduciamo esponenzialmente la size
    # Se ho tutto il buffer (4000$), factor = 1.0. Se ho 200$, factor = 0.05
    safety_factor = max(0.0, min(1.0, dist_ruin / MAX_DAILY_LOSS_AMOUNT))

    # 3. DISTANCE TO TARGET (Soft Landing)
    target_factor = 1.0
    if TRADING_MODE in [1, 2] and current_profit > 0:
        remaining = TARGET_PROFIT_AMOUNT - current_profit
        if remaining <= 0:
            return 0.0  # Target Raggiunto
        # Se manca meno del 10% del target, rallenta
        if remaining < (TARGET_PROFIT_AMOUNT * 0.1):
            target_factor = remaining / (TARGET_PROFIT_AMOUNT * 0.1)

    # 4. CALCOLO RISCHIO MONETARIO
    base_money = equity * BASE_RISK_PER_TRADE
    final_risk = base_money * safety_factor * mode_scaler * target_factor

    # 5. CONVERSIONE IN LOTTI (Stima Euristica)
    # Stima: vol_mult=1.0 -> 1 Std Lot. Usiamo un divisore standard (es. 500$ a lotto per buffer)
    raw_lot = (final_risk / 500.0) * vol_mult

    # 6. CONSISTENCY CHECK (Cap 49%)
    if TRADING_MODE in [1, 2]:
        # Stima profitto scenario "Win" (5R)
        potential_win = final_risk * 5
        if potential_win > MAX_SINGLE_TRADE_PROFIT:
            ratio = MAX_SINGLE_TRADE_PROFIT / potential_win
            raw_lot = raw_lot * ratio

    # 7. NORMALIZZAZIONE BROKER
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0

    step = info.volume_step
    lot = round(raw_lot / step) * step

    if lot < info.volume_min:
        lot = info.volume_min
    if lot > info.volume_max:
        lot = info.volume_max

    # Dead Zone: Se il safety factor è critico (<10%), non aprire per non bruciare commissioni
    if safety_factor < 0.1:
        return 0.0

    return lot

# ==================================================================================
# LOGICA OPERATIVA
# ==================================================================================


class PairHandler:
    def __init__(self, a, b, mult):
        self.a = a
        self.b = b
        self.mult = mult
        self.kf = OnlineKalmanFilter()
        self.residuals = []
        self.active_trade = 0  # 0=Flat, 1=Long Spread, -1=Short Spread

    def on_tick(self):
        # 1. DATI
        tA = mt5.symbol_info_tick(self.a)
        tB = mt5.symbol_info_tick(self.b)
        if tA is None or tB is None:
            return

        # Prezzi Mid
        pA = (tA.bid + tA.ask) / 2
        pB = (tB.bid + tB.ask) / 2

        # 2. AGGIORNAMENTO MODELLO
        err, _ = self.kf.update(pA, pB)
        self.residuals.append(err)
        if len(self.residuals) > LOOKBACK_WINDOW:
            self.residuals.pop(0)

        if len(self.residuals) < 30:
            return  # Riscaldamento

        # Statistiche
        series = np.array(self.residuals)
        mu = np.mean(series)
        sigma = np.std(series)
        if sigma == 0:
            return
        z = (series[-1] - mu) / sigma

        # Filtri Qualità [cite: 15, 12]
        hl = calculate_half_life(series)
        is_clean = check_ljung_box(series)

        # Log Monitor
        # print(f"[{self.a}/{self.b}] Z:{z:.2f} HL:{hl:.1f} Clean:{is_clean}")

        # 3. MACCHINA A STATI
        if self.active_trade == 0:
            # ENTRY LOGIC
            if is_clean and hl <= MAX_HALF_LIFE and abs(z) > Z_ENTRY:

                la = get_smart_lot_size(self.a, self.mult)
                lb = get_smart_lot_size(self.b, self.mult)

                if la > 0 and lb > 0:
                    if z > Z_ENTRY:  # SHORT SPREAD (Sell A / Buy B)
                        print(
                            f"📉 ENTRY SHORT: {self.a}({la}) / {self.b}({lb})")
                        if send_order(self.a, 1, la) and send_order(self.b, 0, lb):
                            self.active_trade = -1
                    elif z < -Z_ENTRY:  # LONG SPREAD (Buy A / Sell B)
                        print(f"📈 ENTRY LONG: {self.a}({la}) / {self.b}({lb})")
                        if send_order(self.a, 0, la) and send_order(self.b, 1, lb):
                            self.active_trade = 1

        else:
            # EXIT LOGIC (Mean Reversion con Isteresi)
            close = False
            if self.active_trade == -1 and z < Z_EXIT:
                close = True
            if self.active_trade == 1 and z > -Z_EXIT:
                close = True
            if abs(z) > 4.5:
                close = True  # Hard Stop Loss statistico

            if close:
                print(f"🔄 EXIT {self.a}/{self.b} (Mean Reversion)")
                close_symbol_positions(self.a)
                close_symbol_positions(self.b)
                self.active_trade = 0

# ==================================================================================
# UTILITY E LOOP PRINCIPALE
# ==================================================================================


def send_order(symbol, type_op, vol):
    # type_op: 0=Buy, 1=Sell
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if type_op == 0 else tick.bid
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": vol,
        "type": mt5.ORDER_TYPE_BUY if type_op == 0 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "magic": 123456,
        "comment": "MDL Quant Pro",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    return res.retcode == mt5.TRADE_RETCODE_DONE


def close_symbol_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            type_op = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(
                symbol).bid if type_op == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": type_op,
                "position": pos.ticket,
                "price": price,
                "magic": 123456
            }
            mt5.order_send(req)


def close_all_panic():
    positions = mt5.positions_get()
    if positions:
        for pos in positions:
            close_symbol_positions(pos.symbol)


def check_global_health():
    """Il Guardiano: Controlla Target e Drawdown"""
    account = mt5.account_info()
    if not account:
        return False

    equity = account.equity
    profit = equity - ACCOUNT_SIZE

    # Check Drawdown
    dd_amount = ACCOUNT_SIZE - equity
    if dd_amount >= MAX_TOTAL_LOSS_AMOUNT:
        print(f"⛔ MAX TOTAL LOSS HIT: -{dd_amount:.2f}. STOP TRADING.")
        close_all_panic()
        return False

    # Check Daily (Semplificato)
    if dd_amount >= MAX_DAILY_LOSS_AMOUNT:
        print(f"⛔ MAX DAILY LOSS HIT. STOP TRADING.")
        close_all_panic()
        return False

    # Check Target
    if TRADING_MODE != 3 and profit >= TARGET_PROFIT_AMOUNT:
        print(f"🎉 TARGET RAGGIUNTO: +{profit:.2f}. STOP TRADING.")
        close_all_panic()  # Chiudi tutto per sicurezza e incassa
        return False

    return True


def main():
    if not mt5.initialize():
        return
    if not mt5.login(FTMO_ACCOUNT_ID, FTMO_PASSWORD, FTMO_SERVER):
        print("❌ Login Fallito")
        return

    print(f"🚀 FINANCE PRO DASHBOARD AVVIATO")
    print(f"📊 Conto: {ACCOUNT_SIZE} | Mode: {TRADING_MODE}")
    print(
        f"🎯 Target: {TARGET_PROFIT_AMOUNT} | 🛡️ Max Loss: {MAX_TOTAL_LOSS_AMOUNT}")

    handlers = [PairHandler(p[0], p[1], p[2]) for p in PAIRS]

    # Warmup
    print("⏳ Caricamento dati e training filtri...")
    for h in handlers:
        rA = mt5.copy_rates_from_pos(h.a, TIMEFRAME, 0, 100)
        rB = mt5.copy_rates_from_pos(h.b, TIMEFRAME, 0, 100)
        if rA is not None and rB is not None:
            for i in range(len(rA)):
                h.kf.update(rA[i]['close'], rB[i]['close'])

    print("✅ Sistema Operativo. In attesa di segnali...")

    while True:
        if not check_global_health():
            time.sleep(60)
            continue

        for h in handlers:
            h.on_tick()

        time.sleep(10)  # Tick ogni 10 sec per non sovraccaricare


if __name__ == "__main__":
    main()
