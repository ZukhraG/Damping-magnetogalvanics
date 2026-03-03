import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from scipy.signal import detrend
import os

# --- Параметры системы ---
K2, xi, gamma, alpha, M0 = 2e4, 4e-4, 1.76e7, 0.02, 780
tn = 150
H_scale = np.sqrt(K2 / xi)
ctau = gamma * H_scale 

def ode_system(t, y, h_fixed, alpha_val):
    phi, dphi = y
    ddphidt = np.sin(4*phi) - h_fixed**2 * np.cos(2*phi) - alpha_val * dphi
    return [dphi, ddphidt]

h_list = np.linspace(1.7, 3.7, 10) 
results_dict = {}
freq_summary = []

if not os.path.exists('data_output_final'):
    os.makedirs('data_output_final')

for h in h_list:
    t_eval = np.linspace(0, tn, 10000)
    sol = solve_ivp(ode_system, [0, tn], [np.pi/4, 0],
                    args=(h, alpha), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    
    t_ns = 1e9 * sol.t / ctau
    phi, dphi = sol.y[0], sol.y[1]
    H_phys = h * H_scale
    
    # 1. Расчет сигналов
    phi_deg = np.degrees(phi)
    beta_deg = np.degrees((xi * H_phys * (np.sin(phi - np.pi/4))) / (2 * M0))
    current_norm = h * np.sin(3*phi + np.pi/4) * dphi
    
    # 2. Частотный анализ
    half = len(phi) // 2
    dt_sec = (t_ns[1] - t_ns[0]) * 1e-9 
    
    def get_spectrum(signal, dt):
        sig_d = detrend(signal[half:])
        N = len(sig_d)
        yf = fft(sig_d)
        xf = fftfreq(N, dt)
        amp = (2.0 / N) * np.abs(yf) # Нормировка 2/N
        return xf[:N//4], amp[:N//4]

    xf, yf_amp = get_spectrum(current_norm, dt_sec)
    
    # Поиск пиковых частот
    f_phi = xf[np.argmax(get_spectrum(phi, dt_sec)[1])] / 1e12
    f_beta = xf[np.argmax(get_spectrum(beta_deg, dt_sec)[1])] / 1e12
    f_j = xf[np.argmax(yf_amp)] / 1e12
    
    # 3. Сохранение в словарь для графиков
    results_dict[h] = {
        't': t_ns, 'phi': phi_deg, 'beta': beta_deg, 'j': current_norm,
        'xf': xf / 1e12, 'yf': yf_amp
    }
    freq_summary.append([h, f_phi, f_beta, f_j])

    # --- СОХРАНЕНИЕ ПОФАЙЛОВО ---
    h_str = f"{h:.2f}".replace('.', '_')
    # Файл 1: Динамика углов и тока
    dyn_data = np.column_stack((t_ns, phi_deg, beta_deg, current_norm))
    np.savetxt(f"data_output_final/dyn_h_{h_str}.txt", dyn_data, 
               header="t(ns)\tphi(deg)\tbeta(deg)\tj_norm", fmt='%.6e')
    # Файл 2: Спектр тока
    spec_data = np.column_stack((xf[:len(yf_amp)] / 1e12, yf_amp))
    np.savetxt(f"data_output_final/spec_h_{h_str}.txt", spec_data, 
               header="freq(THz)\tamplitude(2/N)", delimiter='\t', fmt='%.6e')

# Файл 3: Сводная таблица частот

# --- ВИЗУАЛИЗАЦИЯ ---
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
plot_h = [h_list[0], h_list[4], h_list[-1]]
colors = ['blue', 'green', 'red']

for i, h_val in enumerate(plot_h):
    d = results_dict[h_val]
    m = d['t'] < 15 # смотрим начальный участок
    
    ax[0,0].plot(d['t'][m], d['phi'][m], color=colors[i], label=f'h={h_val:.2f}')
    ax[0,1].plot(d['t'][m], d['beta'][m], color=colors[i])
    ax[1,0].plot(d['t'][m], d['j'][m], color=colors[i])
    ax[1,1].plot(d['xf'], d['yf'], color=colors[i])

ax[0,0].set_ylabel('phi (deg)'); ax[0,0].legend()
ax[0,1].set_ylabel('beta (deg)')
ax[1,0].set_ylabel('j normalized'); ax[1,0].set_xlabel('Time (ns)')
ax[1,1].set_ylabel('Spectral Amplitude (2/N)'); ax[1,1].set_xlabel('Frequency (THz)')
ax[1,1].set_xlim(0, 0.6)

for a in ax.flat: a.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Готово. Данные в папке 'data_output_final'.")