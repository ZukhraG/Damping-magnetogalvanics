import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# --- Параметры системы ---
K2 = 2e4
xi = 4e-4
gamma = 1.76e7
M0 = 800  
tn = 400

ctau = gamma * np.sqrt(K2 / xi)

def ode_system(t, y, h_fixed, alpha):
    phi, dphi = y
    ddphidt = np.sin(4*phi) - h_fixed**2 * np.cos(2*phi) - alpha * dphi
    return [dphi, ddphidt]

h = 2.59
alpha_values = [0.01, 0.02, 0.05, 0.1]
colors = ['blue', 'green', 'red', 'purple']

plt.figure(figsize=(14, 8))

for idx, alpha in enumerate(alpha_values):
    t_eval = np.linspace(0, tn, 100000)
    sol = solve_ivp(ode_system, [0, tn], [np.pi/4, 0], 
                    args=(h, alpha), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    
    t_ns = 1e9 * sol.t / ctau
    phi = sol.y[0]
    
    # Рисуем φ(t) полупрозрачной линией
    plt.plot(t_ns, phi, color=colors[idx], linewidth=0.3, alpha=0.5)
    
    # Находим локальные максимумы и строим огибающую
    peaks, _ = find_peaks(phi, distance=100)
    if len(peaks) > 3:
        f = interp1d(t_ns[peaks], phi[peaks], kind='cubic', fill_value='extrapolate')
        t_smooth = np.linspace(t_ns[0], t_ns[-1], 1000)
        envelope_smooth = f(t_smooth)
        plt.plot(t_smooth, envelope_smooth, color=colors[idx], linewidth=2, 
                label=f'α = {alpha} (огибающая)')

plt.xlabel('Время (нс)', fontsize=12)
plt.ylabel('φ, рад', fontsize=12)
plt.title(f'Затухание ТГц осцилляций при h = {h}', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()