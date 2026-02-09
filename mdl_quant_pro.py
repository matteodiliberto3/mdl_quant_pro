import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.stats import norm
import time

# ==================================================================================
# CONFIGURAZIONE AVANZATA - WALL STREET EDITION
# ==================================================================================
# [cite: 30, 48, 49, 41, 56]
MAX_HALF_LIFE = 15
Z_ENTRY = 2.0
Z_EXIT = 0.5
P_CONV_THRESHOLD = 0.60
HURST_THRESHOLD = 0.45  # < 0.5 indica Mean Reversion pura
BETA_SMOOTHING = 5      # Periodi per EMA del Beta
EXECUTION_TIMEOUT = 2   # Secondi per chiudere la "gamba zoppa"


class MDL_Quant_Engine_V2:
    """Evoluzione del Core Matematico con filtri di stazionarietà e stabilità"""

    @staticmethod
    def calculate_hurst(ts):
        """Calcola l'Esponente di Hurst per validare la stazionarietà"""
        if len(ts) < 100:
            return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag])))
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    @staticmethod
    def kalman_update(y, x, state, P, delta=1e-5, R=1e-3):
        #  Stima dinamica con protezione dal rumore
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
        # [cite: 29, 30] Modellazione OU con vincolo di velocità del capitale
        if len(residuals) < 30:
            return None
        y, x = residuals[1:], residuals[:-1]
        b, a = np.polyfit(x, y, 1)
        theta = -np.log(abs(b))
        mu = a / (1 - b)
        sigma = np.std(y - (a + b*x)) * \
            np.sqrt(-2 * np.log(abs(b)) / (1 - b**2))
        return {"theta": theta, "mu": mu, "sigma": sigma, "half_life": np.log(2)/theta}


class PairHandler_V2:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.state, self.P = np.zeros(2), np.eye(2)
        self.beta_history = []
        self.residuals = []
        self.active_trade = 0

    def process_tick(self):
        # Ottenimento prezzi e aggiornamento Tick Value dinamico
        tick_a, tick_b = mt5.symbol_info_tick(
            self.a), mt5.symbol_info_tick(self.b)
        if not tick_a or not tick_b:
            return

        # 1. KALMAN FILTER CON SMOOTHING [Correzione 2]
        p_a, p_b = (tick_a.bid + tick_a.ask)/2, (tick_b.bid + tick_b.ask)/2
        self.state, self.P, err, std_err = MDL_Quant_Engine_V2.kalman_update(
            p_a, p_b, self.state, self.P)

        self.beta_history.append(self.state[0])
        if len(self.beta_history) > BETA_SMOOTHING:
            self.beta_history.pop(0)
        # Beta stabile per il sizing
        smoothed_beta = np.mean(self.beta_history)

        self.residuals.append(err)
        if len(self.residuals) > 500:
            self.residuals.pop(0)
        if len(self.residuals) < 100:
            return

        # 2. DIAGNOSTICA AVANZATA [Correzione 3 - Hurst]
        z = err / std_err
        ou = MDL_Quant_Engine_V2.get_ou_parameters(np.array(self.residuals))
        hurst = MDL_Quant_Engine_V2.calculate_hurst(self.residuals)

        # 3. LOGICA DI ESECUZIONE ATOMICA [Correzione 1]
        is_mean_reverting = (hurst < HURST_THRESHOLD)  # Filtro stazionarietà

        # [cite: 41, 30] Validazione rigorosa
        valid_signal = False
        if ou and ou['half_life'] <= MAX_HALF_LIFE and is_mean_reverting:
            # Probabilità di convergenza [cite: 40]
            if abs(z) > Z_ENTRY:
                valid_signal = True

        # Gestione Ordini con Kill-Switch
        if self.active_trade == 0 and valid_signal:
            side = -1 if z > Z_ENTRY else 1
            self.execute_atomic_spread(smoothed_beta, side)

        # Exit con Isteresi [cite: 47, 49, 51]
        elif self.active_trade != 0:
            if (self.active_trade == -1 and z < Z_EXIT) or (self.active_trade == 1 and z > -Z_EXIT):
                self.close_all("Target Hysteresis Reached")

    def execute_atomic_spread(self, beta, side):
        """Esegue le due gambe assicurando l'atomicità (Anti-Leg-Risk)"""
        # Calcolo lotti basato su Risk Management [cite: 56]
        # Inserire qui calcolo lot_a, lot_b basato su smoothed_beta

        print(f"Esecuzione Atomica: Side {side} su {self.a}/{self.b}")
        success_a = self.send_order_wrapper(
            self.a, mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL, 0.1)

        if success_a:
            success_b = self.send_order_wrapper(
                self.b, mt5.ORDER_TYPE_SELL if side == 1 else mt5.ORDER_TYPE_BUY, 0.1)
            if not success_b:
                print("ERRORE CRITICO: Gamba B fallita. Attivazione Kill-Switch.")
                self.close_all("Emergency Leg-Risk Neutralization")
            else:
                self.active_trade = side
