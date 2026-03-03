import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры
h = 2.59
alpha = -0.001  

def ode_system(t, y):
    phi, dphi = y
    ddphidt = np.sin(4*phi) - h**2 * np.cos(2*phi) - alpha * dphi
    return [dphi, ddphidt]

# Решаем
t_span = [0, 100]
t_eval = np.linspace(0, 100, 10000)
sol = solve_ivp(ode_system, t_span, [0.1, 0], t_eval=t_eval)

# Смотрим
plt.figure(figsize=(12, 4))

# Временная зависимость
plt.subplot(1, 3, 1)
plt.plot(sol.t, sol.y[0], label=f'α = {alpha},h={h}')
plt.xlabel('t')
plt.ylabel('φ')
plt.title('φ(t)')
plt.legend()  

# Фазовый портрет
plt.subplot(1, 3, 2)
plt.plot(sol.y[0], sol.y[1], label=f'α = {alpha}')
plt.xlabel('φ')
plt.ylabel('dφ/dt')
plt.title('Фазовый портрет')
plt.legend()  

plt.subplot(1, 3, 3)
# Берём последнюю треть и смотрим
n = len(sol.y[0])
last = sol.y[0][2*n//3:]
plt.plot(last, 'b-', label=f'α = {alpha}')
plt.xlabel('шаг')
plt.ylabel('φ')
plt.title('Конец траектории')
plt.legend()  


plt.tight_layout()
plt.show()

# Проверка: амплитуда растёт или постоянна?
amp_start = max(abs(sol.y[0][:1000]))
amp_end = max(abs(sol.y[0][-1000:]))
print(f"Амплитуда в начале: {amp_start:.3f}")
print(f"Амплитуда в конце: {amp_end:.3f}")
print(f"Отношение конец/начало: {amp_end/amp_start:.2f}")