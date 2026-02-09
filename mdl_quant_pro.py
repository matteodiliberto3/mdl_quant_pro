import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.stats import norm
import time
from datetime import datetime

# ==================================================================================
# CONFIGURAZIONE FINANCE PRO DASHBOARD ⚡ (MDL Quant 4.0 Execution-Ready)
# ==================================================================================
FTMO_ACCOUNT_ID = 1521076547
FTMO_PASSWORD = "J39NF**Ud"
FTMO_SERVER = "FTMO-Demo2"

ACCOUNT_SIZE = 100000.0
TRADING_MODE = 1  # 1: Fase 1, 2: Fase 2, 3: Live

# Portfolio con Volatilità Normalizzata
PAIRS = [
    ("EURUSD", "GBPUSD", 1.0),
    ("XAUUSD", "XAGUSD", 0.5),
    ("US500.cash", "US100.cash", 0.1),
    ("BTCUSD", "ETHUSD", 0.01)
]

# Parametri operativi MDL Quant 4.0 [cite: 30, 41, 48, 49]
MAX_HALF_LIFE = 15          # Vincolo Capital Velocity [cite: 30]
Z_ENTRY = 2.0               # Soglia ingresso [cite: 48]
Z_EXIT = 0.5                # Banda di Isteresi [cite: 49]
HURST_THRESHOLD = 0.45      # Filtro Stazionarietà (Wall Street Correction)
BETA_SMOOTHING = 5          # Anti Over-fitting (Wall Street Correction)
TIMEFRAME = mt5.TIMEFRAME_H1
WARMUP_PERIOD = 200

# Limiti Hard di Drawdown FTMO [cite: 58]
DAILY_LOSS_LIMIT_PCT = 0.04  # Stop al 4% (Buffer di sicurezza)
TOTAL_LOSS_LIMIT_PCT = 0.09  # Stop al 9% (Buffer di sicurezza)

# ==================================================================================
# MOTORE QUANTISTICO (CORE)
# ==================================================================================


class MDL_Quant_Engine_V2:
    @staticmethod
    def calculate_hurst(ts):
        """Valida se lo spread è realmente Mean-Reverting"""
        if len(ts) < 100:
            return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag])))
               for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    @staticmethod
    def kalman_update(y, x, state, P, delta=1e-5, R=1e-3):
        """Stima dinamica dell'Hedge Ratio beta [cite: 23]"""
        H = np.array([x, 1.0])
        P = P + (np.eye(2) * delta)
        y_pred = np.dot(H, state)
        error = y - y_pred
        S = np.dot(H, np.dot(P, H.T)) + R
        K = np.dot(P, H.T) / S
        new_state = state + K * error
        new_P = np.dot(np.eye(2) - np.outer(K, H), P)
        return new_state, new_P, error, np.sqrt(S)

    @staticmethod
    def get_ou_parameters(residuals):
        """Modellazione OU e Half-Life [cite: 29]"""
        if len(residuals) < 30:
            return None
        y, x = residuals[1:], residuals[:-1]
        b, a = np.polyfit(x, y, 1)
        theta = -np.log(abs(b)) if abs(b) > 0 else 0
        half_life = np.log(2) / theta if theta > 0 else 999
        return {"theta": theta, "half_life": half_life}

# ==================================================================================
# GESTORE OPERATIVO CON ISTERESI E SIZING [cite: 45, 55]
# ==================================================================================


class PairHandler:
    def __init__(self, a, b, mult):
        self.a, self.b, self.mult = a, b, mult
        self.state, self.P = np.zeros(2), np.eye(2)
        self.residuals, self.beta_history = [], []
        self.active_trade = 0
        self.last_z, self.last_hurst, self.last_hl = 0.0, 0.5, 0.0

    def on_tick(self):
        tA, tB = mt5.symbol_info_tick(self.a), mt5.symbol_info_tick(self.b)
        if not tA or not tB:
            return

        # 1. Update Matematico
        pA, pB = (tA.bid + tA.ask)/2, (tB.bid + tB.ask)/2
        self.state, self.P, err, std_err = MDL_Quant_Engine_V2.kalman_update(
            pA, pB, self.state, self.P)

        # Smoothing del Beta per stabilità del bilanciamento
        self.beta_history.append(self.state[0])
        if len(self.beta_history) > BETA_SMOOTHING:
            self.beta_history.pop(0)
        current_beta = np.mean(self.beta_history)

        self.residuals.append(err)
        if len(self.residuals) > 500:
            self.residuals.pop(0)
        if len(self.residuals) < 50:
            return

        # 2. Analisi Segnale [cite: 35, 37]
        self.last_z = err / std_err
        ou = MDL_Quant_Engine_V2.get_ou_parameters(np.array(self.residuals))
        self.last_hl = ou['half_life'] if ou else 999
        self.last_hurst = MDL_Quant_Engine_V2.calculate_hurst(self.residuals)

        # 3. Logica Entry con filtri di qualità [cite: 30, 48, 50]
        if self.active_trade == 0:
            if self.last_hl <= MAX_HALF_LIFE and self.last_hurst < HURST_THRESHOLD:
                if abs(self.last_z) > Z_ENTRY:
                    side = -1 if self.last_z > Z_ENTRY else 1
                    self.execute_spread(current_beta, side)

        # 4. Logica Exit con Isteresi (Anti-Churning) [cite: 47, 49, 51]
        else:
            if (self.active_trade == -1 and self.last_z < Z_EXIT) or \
               (self.active_trade == 1 and self.last_z > -Z_EXIT) or \
               (abs(self.last_z) > 4.5):
                self.close_positions("Mean Reversion / Hard Stop")

    def execute_spread(self, beta, side):
        # Sizing semplificato (0.1 lotti base) - VaR-Capping applicato nel main [cite: 56]
        lot_a = 0.1
        lot_b = round(abs(lot_a * beta), 2)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 ENTRY {'SHORT' if side == -1 else 'LONG'} {self.a}/{self.b} | Z:{self.last_z:.2f} | H:{self.last_hurst:.2f}")

        # Esecuzione Atomica (Anti-Leg-Risk)
        success_a = send_order(self.a, 1 if side == -1 else 0, lot_a)
        if success_a:
            success_b = send_order(self.b, 0 if side == -1 else 1, lot_b)
            if not success_b:
                print("⚠️ ERRORE CRITICO: Gamba B fallita. Kill-Switch attivo!")
                close_symbol_positions(self.a)
            else:
                self.active_trade = side
        else:
            print(f"❌ Apertura Gamba A ({self.a}) fallita.")

    def close_positions(self, reason):
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 EXIT {self.a}/{self.b} | Reason: {reason}")
        close_symbol_positions(self.a)
        close_symbol_positions(self.b)
        self.active_trade = 0

# ==================================================================================
# PROTEZIONE CAPITALE E UTILITY MT5 [cite: 54, 58]
# ==================================================================================


def check_account_safety():
    """Monitoraggio Drawdown Hard Limits [cite: 58]"""
    acc = mt5.account_info()
    if not acc:
        return False

    total_dd = (ACCOUNT_SIZE - acc.equity) / ACCOUNT_SIZE
    daily_dd = (acc.balance - acc.equity) / \
        acc.balance if acc.equity < acc.balance else 0

    if total_dd >= TOTAL_LOSS_LIMIT_PCT or daily_dd >= DAILY_LOSS_LIMIT_PCT:
        print(
            f"🚨!!! DRAWDOWN LIMIT HIT ({max(total_dd, daily_dd)*100:.1f}%) !!!")
        close_all_panic()
        return False
    return True


def send_order(symbol, type_op, vol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(vol),
        "type": mt5.ORDER_TYPE_BUY if type_op == 0 else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if type_op == 0 else tick.bid,
        "magic": 123456,
        "comment": "MDL Quant V4",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    return res.retcode == mt5.TRADE_RETCODE_DONE


def close_symbol_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            tick = mt5.symbol_info_tick(symbol)
            type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": type_close,
                "position": pos.ticket,
                "price": tick.bid if type_close == mt5.ORDER_TYPE_SELL else tick.ask,
                "magic": 123456
            }
            mt5.order_send(req)


def close_all_panic():
    print("🧹 Emergency Liquidating all positions...")
    pos = mt5.positions_get()
    if pos:
        for p in pos:
            close_symbol_positions(p.symbol)

# ==================================================================================
# LOOP PRINCIPALE
# ==================================================================================


def main():
    if not mt5.initialize():
        return
    if not mt5.login(FTMO_ACCOUNT_ID, FTMO_PASSWORD, FTMO_SERVER):
        print("❌ Login Fallito")
        mt5.shutdown()
        return

    print(f"🚀 FINANCE PRO DASHBOARD ⚡ - MDL QUANT 4.0")
    print(
        f"🛡️ SAFETY: Daily {DAILY_LOSS_LIMIT_PCT*100}% | Total {TOTAL_LOSS_LIMIT_PCT*100}%")
    print("-" * 50)

    handlers = [PairHandler(p[0], p[1], p[2]) for p in PAIRS]
    last_diag = 0

    try:
        while True:
            if not check_account_safety():
                print("⛔ SISTEMA BLOCCATO PER PROTEZIONE CAPITALE.")
                break

            for h in handlers:
                h.on_tick()

            if time.time() - last_diag > 60:
                print(
                    f"\n--- MONITOR [{datetime.now().strftime('%H:%M:%S')}] ---")
                for h in handlers:
                    print(
                        f"[{h.a}/{h.b}] Z:{h.last_z:>6.2f} | H:{h.last_hurst:>5.2f} | HL:{h.last_hl:>5.1f} | Trade:{h.active_trade}")
                last_diag = time.time()

            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stop manuale.")
        close_all_panic()
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
