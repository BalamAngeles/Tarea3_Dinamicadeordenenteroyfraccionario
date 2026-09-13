"""
Ajuste no lineal
Reproduce el notebook de Wolfram Mathematica que:
  1. Grafca la velocidad teorica  v(t) = (g*m/k)*(1 - exp(-k*t/m))  con k=0.013, m=0.003, g=9.81
  2. Define datos experimentales d1
  3. Ajusta el mismo modelo no lineal para estimar k
  4. Calcula el coeficiente de determinacion R^2
  5. Superpone la curva ajustada con los puntos experimentales
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Grafica teorica (Pagina inicial del notebook)
# ----------------------------------------------------------------------
k = 0.013
m = 0.003
g = 9.81


def v_teorica(t, k, m, g):
    return (1 - np.exp(-k * t / m)) * g * m / k


t_teorica = np.linspace(0, 4, 200)
v_teorica_vals = v_teorica(t_teorica, k, m, g)

plt.figure(figsize=(8, 5))
plt.plot(t_teorica, v_teorica_vals, label="v(t) teorica")
plt.xlabel("t")
plt.ylabel("v[t]")
plt.title("Velocidad teorica  (k=0.013, m=0.003, g=9.81)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 2. Datos experimentales (d1)
# ----------------------------------------------------------------------
d1 = [
    (0, 0),
    (0.04166666667, 0.3454752631),
    (0.08333333333, 0.5690180805),
    (0.125, 1.097392012),
    (0.1666666667, 1.381901053),
    (0.2083333333, 1.422545201),
    (0.25, 1.483511424),
    (0.2916666667, 1.60544387),
    (0.3333333333, 1.828986687),
    (0.375, 1.788342539),
    (0.4166666667, 1.768020464),
    (0.4583333333, 1.991563282),
    (0.5, 2.03220743),
    (0.5416666667, 1.991563282),
    (0.5833333333, 1.88995291),
    (0.625, 2.194784025),
    (0.6666666667, 2.357360619),
    (0.7083333333, 2.199552035),
    (0.75, 2.099635914),
    (0.7916666667, 1.954307635),
    (0.8333333333, 2.243834692),
    (0.875, 2.243834692),
    (0.9166666667, 2.0266894),
    (0.9583333333, 2.062880282),
    (1, 2.243834692),
    (1.041666667, 2.316216457),
    (1.083333333, 2.062880282),
    (1.125, 2.316216457),
    (1.166666667, 2.424789103),
]

d1 = np.array(d1, dtype=float)
t_data = d1[:, 0]
v_data = d1[:, 1]

# ----------------------------------------------------------------------
# 3. Ajuste no lineal del modelo  v(t) = (1 - exp(-k*t/m)) * g*m/k
#    (en el notebook m y g son fijos y se estima solo k)
# ----------------------------------------------------------------------
def modelo(t, k):
    return (1 - np.exp(-k * t / m)) * g * m / k


# Valor inicial para k (igual al valor teorico como punto de partida)
p0 = [0.013]

popt, pcov = curve_fit(modelo, t_data, v_data, p0=p0, maxfev=10000)
k_ajustado = popt[0]
k_err = np.sqrt(np.diag(pcov))[0]

print("=" * 50)
print("AJUSTE NO LINEAL")
print("=" * 50)
print(f"m = {m}, g = {g} (fijos)")
print(f"k ajustado = {k_ajustado:.12f}")
print(f"k error estandar = {k_err:.4e}")

# ----------------------------------------------------------------------
# 4. Cambio de variable (exp(-k*t/m) = exp(-4.36728...*t)) y R^2
# ----------------------------------------------------------------------
# Prediccion del modelo ajustado
v_pred = modelo(t_data, k_ajustado)

# Coeficiente de determinacion R^2
ss_res = np.sum((v_data - v_pred) ** 2)
ss_tot = np.sum((v_data - np.mean(v_data)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"R^2 = {r2:.15f}")

# Forma equivalente de la funcion ajustada:
# v(t) = (g*m/k)*(1 - exp(-k*t/m))
#     = 2.246246622...*(1 - exp(-4.36728536...*t))
gmk = g * m / k_ajustado
beta = k_ajustado / m
print(f"\nForma equivalente:  v(t) = {gmk:.12f} * (1 - exp({-beta:.12f}*t))")

# ----------------------------------------------------------------------
# 5. Grafica superpuesta: curva ajustada + puntos experimentales
# ----------------------------------------------------------------------
t_fit = np.linspace(0, 1.2, 200)
v_fit = modelo(t_fit, k_ajustado)

plt.figure(figsize=(8, 5))
plt.plot(t_fit, v_fit, "r--", linewidth=2, label="Curva ajustada")
plt.scatter(t_data, v_data, label="Datos experimentales d1")
plt.xlabel("t")
plt.ylabel("v[t]")
plt.title(f"Ajuste no lineal  (k = {k_ajustado:.6f},  R$^2$ = {r2:.4f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nPrograma terminado.")
