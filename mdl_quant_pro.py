import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
# Necessaria: pip install statsmodels
from statsmodels.tsa.stattools import adfuller

# ==================================================================================
# CONFIGURAZIONE DINAMICA FINANCE PRO DASHBOARD ⚡
# ==================================================================================
FTMO_ACCOUNT_ID = 1521076547
FTMO_PASSWORD = "J39NF**Ud"
FTMO_SERVER = "FTMO-Demo2"

PATH_MT5 = None  # Inserire percorso se MT5 non viene rilevato automaticamente

PAIRS = [
    ("EURUSD", "GBPUSD", 1.0),
    ("XAUUSD", "XAGUSD", 0.5),
    ("US500.cash", "US100.cash", 0.1),
    ("BTCUSD", "ETHUSD", 0.01)
]

# Parametri Strategia MDL Quant 4.0
MAX_HALF_LIFE = 15
Z_ENTRY = 2.0
Z_EXIT = 0.5
HURST_THRESHOLD = 0.45
BETA_SMOOTHING = 5
TIMEFRAME = mt5.TIMEFRAME_H1

# Sicurezza FTMO
DAILY_LOSS_LIMIT_PCT = 0.04
TOTAL_LOSS_LIMIT_PCT = 0.09
INITIAL_DYNAMIC_BALANCE = 0.0

# ==================================================================================
# CORE MATEMATICO RAFFINATO
# ==================================================================================


class MDL_Engine:
    @staticmethod
    def calculate_hurst(ts):
        if len(ts) < 50:
            return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag])))
               for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    @staticmethod
    def kalman_update(y, x, state, P, delta=1e-5, R=1e-3):
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
        if len(residuals) < 30:
            return None
        y, x = residuals[1:], residuals[:-1]
        b, a = np.polyfit(x, y, 1)
        theta = -np.log(abs(b)) if abs(b) > 0 else 0
        return {"half_life": np.log(2)/theta if theta > 0 else 999}

    @staticmethod
    def check_cointegration(residuals):
        """Verifica se la relazione tra i due asset è statisticamente stabile."""
        if len(residuals) < 100:
            return False
        try:
            # Test di Dickey-Fuller Aumentato: p-value < 0.05 indica stazionarietà
            result = adfuller(residuals)
            return result[1] < 0.05
        except:
            return False

# ==================================================================================
# GESTORE OPERATIVO CON FILTRI STATISTICI
# ==================================================================================


class PairHandler:
    def __init__(self, a, b, mult):
        self.a, self.b, self.mult = a, b, mult
        self.state, self.P = np.zeros(2), np.eye(2)
        self.residuals, self.beta_hist = [], []
        self.active_trade = 0
        self.z, self.hurst, self.hl = 0, 0.5, 0
        self.is_coint = False
        self.last_coint_check = 0

    def on_tick(self):
        tA, tB = mt5.symbol_info_tick(self.a), mt5.symbol_info_tick(self.b)
        if not tA or not tB:
            return

        pA, pB = (tA.bid + tA.ask)/2, (tB.bid + tB.ask)/2

        # Aggiornamento Filtro di Kalman (Stima del Beta dinamico)
        self.state, self.P, err, std_err = MDL_Engine.kalman_update(
            pA, pB, self.state, self.P)

        self.beta_hist.append(self.state[0])
        if len(self.beta_hist) > BETA_SMOOTHING:
            self.beta_hist.pop(0)

        self.residuals.append(err)
        if len(self.residuals) > 500:
            self.residuals.pop(0)

        if len(self.residuals) < 100:
            return

        # Calcoli Statistici
        self.z = err / std_err
        ou = MDL_Engine.get_ou_parameters(np.array(self.residuals))
        self.hl = ou['half_life'] if ou else 999
        self.hurst = MDL_Engine.calculate_hurst(self.residuals)

        # Verifica Cointegrazione ogni 300 tick per risparmiare CPU
        if time.time() - self.last_coint_check > 300:
            self.is_coint = MDL_Engine.check_cointegration(self.residuals)
            self.last_coint_check = time.time()

        # LOGICA DI INGRESSO (Mean Reversion pura)
        if self.active_trade == 0:
            if self.is_coint and self.hl <= MAX_HALF_LIFE and self.hurst < HURST_THRESHOLD:
                if abs(self.z) > Z_ENTRY:
                    self.execute(np.mean(self.beta_hist), -
                                 1 if self.z > 0 else 1)

        # LOGICA DI USCITA
        else:
            # Uscita a target (Z_EXIT) o per sicurezza se il trade diverge troppo (Z > 4.5)
            if (self.active_trade == -1 and self.z < Z_EXIT) or \
               (self.active_trade == 1 and self.z > -Z_EXIT) or abs(self.z) > 4.5:
                self.close()

    def execute(self, beta, side):
        lot = 0.05  # Gestione lotti base
        if self.order(self.a, 1 if side == -1 else 0, lot):
            # Il lotto della gamba B è pesato sul Beta calcolato da Kalman
            lot_b = abs(lot * beta)
            if self.order(self.b, 0 if side == -1 else 1, lot_b):
                self.active_trade = side
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ✅ ENTRY {self.a}/{self.b} | Z: {self.z:.2f} | Beta: {beta:.2f}")
            else:
                self.close()

    def order(self, sym, typ, vol):
        t = mt5.symbol_info_tick(sym)
        # Protezione volumi: minimo 0.01 lotti
        vol = max(float(round(vol, 2)), 0.01)
        r = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": vol,
            "type": mt5.ORDER_TYPE_BUY if typ == 0 else mt5.ORDER_TYPE_SELL,
            "price": t.ask if typ == 0 else t.bid,
            "magic": 999,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "deviation": 10
        }
        res = mt5.order_send(r)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Errore Ordine su {sym}: {res.comment}")
        return res.retcode == mt5.TRADE_RETCODE_DONE

    def close(self):
        for s in [self.a, self.b]:
            positions = mt5.positions_get(symbol=s)
            if positions:
                for p in positions:
                    if p.magic == 999:
                        t = mt5.symbol_info_tick(s)
                        type_close = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        price_close = t.bid if type_close == mt5.ORDER_TYPE_SELL else t.ask
                        mt5.order_send({
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": s,
                            "volume": p.volume,
                            "type": type_close,
                            "position": p.ticket,
                            "price": price_close,
                            "magic": 999,
                            "type_filling": mt5.ORDER_FILLING_IOC
                        })
        self.active_trade = 0
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 TARGET/EXIT {self.a}/{self.b}")

# ==================================================================================
# MAIN ENGINE CON FIX CONNESSIONE
# ==================================================================================


def main():
    global INITIAL_DYNAMIC_BALANCE
    print(f"--- FINANCE PRO DASHBOARD ⚡ INITIALIZING ---")

    # Inizializzazione robusta
    init_params = {"server": FTMO_SERVER,
                   "login": FTMO_ACCOUNT_ID, "password": FTMO_PASSWORD}
    if PATH_MT5:
        init_params["path"] = PATH_MT5

    if not mt5.initialize(**init_params):
        print(
            f"❌ ERRORE CRITICO: Inizializzazione fallita. Codice errore: {mt5.last_error()}")
        return

    # Verifica stato connessione
    acc = mt5.account_info()
    if acc is None:
        print(
            f"❌ LOGIN FALLITO: Credenziali errate o server {FTMO_SERVER} non raggiungibile.")
        mt5.shutdown()
        return

    INITIAL_DYNAMIC_BALANCE = acc.balance
    print(
        f"✅ CONNESSO: {acc.name} | Conto: {INITIAL_DYNAMIC_BALANCE} {acc.currency}")

    handlers = [PairHandler(p[0], p[1], p[2]) for p in PAIRS]

    try:
        while True:
            # Controllo Equity per Sicurezza FTMO
            curr = mt5.account_info()
            if curr:
                drawdown = (INITIAL_DYNAMIC_BALANCE -
                            curr.equity) / INITIAL_DYNAMIC_BALANCE
                if drawdown >= TOTAL_LOSS_LIMIT_PCT:
                    print(
                        f"🚨 EMERGENCY STOP: Drawdown raggiunto ({drawdown*100:.2f}%)!")
                    break

            for h in handlers:
                h.on_tick()

            time.sleep(1)  # Rispetto del rate-limit
    except KeyboardInterrupt:
        print("🛑 Script interrotto dall'utente.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
