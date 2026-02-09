import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.stats import chi2, norm
import time
from datetime import datetime

# ==================================================================================
# CONFIGURAZIONE FINANCE PRO DASHBOARD ⚡ (MDL Quant 4.0 Edition)
# ==================================================================================
FTMO_ACCOUNT_ID = 1521076547
FTMO_PASSWORD = "J39NF**Ud"
FTMO_SERVER = "FTMO-Demo2"

ACCOUNT_SIZE = 100000.0
TRADING_MODE = 1  # 1: Fase 1, 2: Fase 2, 3: Live

PAIRS = [
    ("EURUSD", "GBPUSD", 1.0),
    ("XAUUSD", "XAGUSD", 0.5),
    ("US500", "US100", 0.1),
    ("BTCUSD", "ETHUSD", 0.01)
]

# Parametri dal PDF e Correzioni Wall Street
MAX_HALF_LIFE = 15          # [cite: 30]
Z_ENTRY = 2.0               # [cite: 48]
Z_EXIT = 0.5                # Banda Isteresi [cite: 49]
HURST_THRESHOLD = 0.45      # Correzione: Filtro Mean-Reversion
BETA_SMOOTHING = 5          # Correzione: Anti Over-fitting
TIMEFRAME = mt5.TIMEFRAME_H1
WARMUP_PERIOD = 200

# ==================================================================================
# MOTORE QUANTISTICO AVANZATO
# ==================================================================================


class MDL_Quant_Engine_V2:
    @staticmethod
    def calculate_hurst(ts):
      #  """Valida se lo spread è realmente Mean-Reverting"""
        if len(ts) < 100:
            return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag])))
               for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    @staticmethod
    def kalman_update(y, x, state, P, delta=1e-5, R=1e-3):
        # "Stima dinamica dell'Hedge Ratio [cite: 22, 23]"
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
        # "Modellazione OU per Capital Velocity [cite: 29]"
        if len(residuals) < 30:
            return None
        y, x = residuals[1:], residuals[:-1]
        b, a = np.polyfit(x, y, 1)
        theta = -np.log(abs(b))
        half_life = np.log(2) / theta if theta > 0 else 999
        return {"theta": theta, "half_life": half_life}

# ==================================================================================
# GESTORE OPERATIVO
# ==================================================================================


class PairHandler:
    def __init__(self, a, b, mult):
        self.a, self.b, self.mult = a, b, mult
        self.state, self.P = np.zeros(2), np.eye(2)
        self.residuals, self.beta_history = [], []
        self.active_trade = 0

    def on_tick(self):
        tA, tB = mt5.symbol_info_tick(self.a), mt5.symbol_info_tick(self.b)
        if not tA or not tB:
            return

        # 1. Update Matematico
        pA, pB = (tA.bid + tA.ask)/2, (tB.bid + tB.ask)/2
        self.state, self.P, err, std_err = MDL_Quant_Engine_V2.kalman_update(
            pA, pB, self.state, self.P)

        # Correzione: Smoothing Beta
        self.beta_history.append(self.state[0])
        if len(self.beta_history) > BETA_SMOOTHING:
            self.beta_history.pop(0)
        current_beta = np.mean(self.beta_history)

        self.residuals.append(err)
        if len(self.residuals) > 500:
            self.residuals.pop(0)
        if len(self.residuals) < 50:
            return

        # 2. Analisi Segnale
        z = err / std_err
        ou = MDL_Quant_Engine_V2.get_ou_parameters(np.array(self.residuals))
        hurst = MDL_Quant_Engine_V2.calculate_hurst(self.residuals)

        # 3. Logica Entry
        if self.active_trade == 0:
            if ou and ou['half_life'] <= MAX_HALF_LIFE and hurst < HURST_THRESHOLD:
                if abs(z) > Z_ENTRY:
                    side = -1 if z > Z_ENTRY else 1
                    self.execute_spread(current_beta, side)

        # [cite_start]4. Logica Exit con Isteresi [cite: 47, 51]
        else:
            if (self.active_trade == -1 and z < Z_EXIT) or \
               (self.active_trade == 1 and z > -Z_EXIT) or \
               (abs(z) > 4.5):
                self.close_positions("Mean Reversion / Hard Stop")

    def execute_spread(self, beta, side):
        lot_a = 0.1  # Semplificato per l'esempio, usa get_smart_lot_size qui
        lot_b = abs(lot_a * beta)

        print(f"[{datetime.now()}] 🚀 ENTRY {'SHORT' if side == -1 else 'LONG'} {self.a}/{self.b} | Z:{Z_ENTRY} | Hurst:{HURST_THRESHOLD}")

        # Correzione: Esecuzione Atomica (Kill-Switch)
        success_a = send_order(self.a, 1 if side == -1 else 0, lot_a)
        if success_a:
            success_b = send_order(self.b, 0 if side == -1 else 1, lot_b)
            if not success_b:
                print("⚠️ ERRORE CRITICO: Gamba B fallita. Kill-Switch attivo!")
                close_symbol_positions(self.a)
            else:
                self.active_trade = side
        else:
            print("❌ Apertura Gamba A fallita.")

    def close_positions(self, reason):
        print(f"[{datetime.now()}] 🔄 EXIT {self.a}/{self.b} Reason: {reason}")
        close_symbol_positions(self.a)
        close_symbol_positions(self.b)
        self.active_trade = 0

# ==================================================================================
# UTILITY MT5
# ==================================================================================


def send_order(symbol, type_op, vol):
    tick = mt5.symbol_info_tick(symbol)
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
            price = tick.bid if type_close == mt5.ORDER_TYPE_SELL else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": type_close,
                "position": pos.ticket,
                "price": price,
                "magic": 123456
            }
            mt5.order_send(req)


def main():
    if not mt5.initialize():
        return
    if not mt5.login(FTMO_ACCOUNT_ID, FTMO_PASSWORD, FTMO_SERVER):
        print("❌ Login Fallito")
        return

    print(f"🚀 FINANCE PRO DASHBOARD ⚡ - MDL QUANT 4.0")
    print(f"Monitoraggio attivo su {len(PAIRS)} coppie...")

    handlers = [PairHandler(p[0], p[1], p[2]) for p in PAIRS]

    while True:
        for h in handlers:
            h.on_tick()
        time.sleep(1)  # Monitoraggio continuo ogni secondo


if __name__ == "__main__":
    main()
