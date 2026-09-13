# -*- coding: utf-8 -*-
"""
Comparacion de modelos matematicos vs datos experimentales
Filtro de cafe + 3 clips (Pruebas 3.1, 3.2 y 3.3) - CASO 3 CLIPS.

Genera UNA SOLA grafica con:
  - Datos experimentales vbar +/- dv (agrupados por tolerancia, d.a.m.)
  - Caso 1: caida libre            v(t) = -g t
  - Caso 2: arrastre lineal        v(t) = v_t [1 - exp(-(b/m) t)],  b = m g/|v_t|
  - Caso 3: arrastre de Newton     v(t) = v_t tanh(g/v_t t)
  - Linea de velocidad terminal v_t

Ademas calcula metricas cuantitativas para comparar los modelos:
  RMSE, R^2 y porcentaje de barras de error intersectadas por cada curva.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. DATOS EXPERIMENTALES (t, v) de las 3 pruebas (caso filtro + 3 clips)
# ---------------------------------------------------------------------------
prueba1 = np.array([
    (0.000, -0.66666667), (0.033, -0.77272727), (0.066, -0.92537313),
    (0.100, -1.04477612), (0.133, -1.16666667), (0.166, -1.05970149),
    (0.200, -1.23880597), (0.233, -1.45454545), (0.266, -1.35820896),
    (0.300, -1.47761194), (0.333, -1.68181818), (0.366, -1.77611940),
    (0.400, -1.80597015), (0.433, -1.86363636), (0.466, -1.85074627),
    (0.500, -1.83582090), (0.533, -1.80303030), (0.566, -1.76119403),
    (0.600, -1.85074627), (0.633, -1.98484848), (0.666, -1.97014925),
    (0.700, -1.97014925), (0.733, -2.06060606), (0.766, -2.07462687),
    (0.800, -1.91044776), (0.833, -1.90909091), (0.866, -2.06060606),
    (0.899, -2.05970149), (0.933, -2.02985075), (0.966, -2.00000000),
    (0.999, -1.95522388), (1.033, -1.95522388), (1.066, -1.96969697),
    (1.099, -1.89686567), (1.133, -1.43388060), (1.166, -0.41469697),
    (1.199, 0.03510292),
])

prueba2 = np.array([
    (0.000, -0.06060606), (0.033, -0.45454545), (0.066, -0.95522388),
    (0.100, -1.01492537), (0.133, -0.92424242), (0.166, -1.01492537),
    (0.200, -1.08955224), (0.233, -1.10606061), (0.266, -1.16417910),
    (0.300, -1.38805970), (0.333, -1.51515152), (0.366, -1.52238806),
    (0.400, -2.11940299), (0.433, -2.18181818), (0.466, -1.83582090),
    (0.500, -1.94029851), (0.533, -2.00000000), (0.566, -1.77272727),
    (0.599, -2.02985075), (0.633, -2.10447761), (0.666, -1.80303030),
    (0.699, -1.86567164), (0.733, -1.74626866), (0.766, -1.77272727),
    (0.799, -2.53731343), (0.833, -2.62686567), (0.866, -2.03030303),
    (0.899, -2.00000000), (0.933, -1.98507463), (0.966, -2.01515152),
    (0.999, -2.00000000), (1.033, -1.98507463), (1.066, -2.01515152),
    (1.099, -1.97014925), (1.133, -1.95447761), (1.166, -1.46196970),
    (1.199, 0.01548027),
])

prueba3 = np.array([
    (0.000, -0.55882353), (0.034, -0.65671642), (0.067, -0.66666667),
    (0.100, -0.53731343), (0.134, -0.64179104), (0.167, -0.86363636),
    (0.200, -0.91044776), (0.234, -1.29850746), (0.267, -1.59090909),
    (0.300, -1.46268657), (0.334, -1.59701493), (0.367, -1.84848485),
    (0.400, -1.83582090), (0.434, -1.74626866), (0.467, -1.77272727),
    (0.500, -1.74626866), (0.534, -1.79104478), (0.567, -1.83333333),
    (0.600, -1.77611940), (0.634, -1.73134328), (0.667, -1.83333333),
    (0.700, -1.92424242), (0.733, -1.88059701), (0.767, -2.16417910),
    (0.800, -2.18181818), (0.833, -1.86567164), (0.867, -1.89552239),
    (0.900, -1.96969697), (0.933, -1.94029851), (0.967, -1.91044776),
    (1.000, -2.00000000), (1.033, -1.97014925), (1.067, -2.01044776),
    (1.100, -1.92424242), (1.133, -1.97014925), (1.167, -1.74328358),
    (1.200, 0.08740360),
])

# ---------------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------------
TOL_AGRUPACION = 0.01     # s, agrupar tiempos practicamente iguales
T_MIN          = 0.10     # excluir puntos iniciales ruidosos de las metricas
T_MAX          = 1.17     # limite de validez (objeto toca el suelo)

# Parametros del sistema (Caso filtro + 3 clips)
m_FILTRO = 0.00093        # kg
m_CLIP   = 0.00054       # kg
N_CLIPS  = 3
g        = 9.81           # m/s^2

# Velocidad terminal experimental (estimada de la fase estacionaria de los
# datos, t ~ 0.9-1.17 s; se puede forzar con VT_EXP = None para usar la de
# datos, o fijarla manualmente):
VT_EXP = -1.95            # m/s  (valor del documento de comparacion)

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, "comparacion_modelos_3clips.png")

# ---------------------------------------------------------------------------
# 3. TRATAMIENTO ESTADISTICO: agrupado por tolerancia (promedio + d.a.m.)
# ---------------------------------------------------------------------------
def dam(valores):
    xbar = np.mean(valores)
    return xbar, np.max(np.abs(valores - xbar))

pares = np.concatenate([prueba1, prueba2, prueba3])
pares = pares[np.argsort(pares[:, 0])]

grupos = []
for t_i, v_i in pares:
    if grupos and abs(t_i - np.mean(grupos[-1][0])) <= TOL_AGRUPACION:
        grupos[-1][0].append(t_i)
        grupos[-1][1].append(v_i)
    else:
        grupos.append(([t_i], [v_i]))

t     = []
v_bar = []
dv    = []
for ts, vs in grupos:
    t_prom = np.mean(ts)
    vb, dvv = dam(np.array(vs))
    if dvv == 0.0:
        dvv = 0.001
    t.append(t_prom)
    v_bar.append(vb)
    dv.append(dvv)
t, v_bar, dv = map(np.array, (t, v_bar, dv))

# ---------------------------------------------------------------------------
# 4. PARAMETROS DE LOS MODELOS
# ---------------------------------------------------------------------------
m = m_FILTRO + N_CLIPS * m_CLIP

# Velocidad terminal estimada a partir de la fase estacionaria de los datos
fase_estacionaria = (t >= 0.9) & (t <= T_MAX)
VT_DATOS = float(np.mean(v_bar[fase_estacionaria]))

vt = VT_EXP if VT_EXP is not None else VT_DATOS
vt_mag = abs(vt)

# Coeficiente de arrastre lineal (Caso 2):  v_t = -mg/b  ->  b = mg/|v_t|
b_caso2 = m * g / vt_mag
p_caso2 = b_caso2 / m          # = g/|v_t|

print("=" * 78)
print("PARAMETROS DEL SISTEMA  (filtro de cafe + 3 clips)")
print("=" * 78)
print(f"  m = m_filtro + 3*m_clip = {m:.5f} kg")
print(f"  v_t (constante del documento) = {vt:.4f} m/s")
print(f"  v_t (estimada de los datos)   = {VT_DATOS:.4f} m/s")
print(f"  b (arrastre lineal)     = {b_caso2:.5f} kg/s")
print(f"  b/m (exponente Caso 2)  = {p_caso2:.4f} s^-1")
print("=" * 78)

# ---------------------------------------------------------------------------
# 5. FUNCIONES DE LOS MODELOS y evaluacion sobre puntos experimentales
# ---------------------------------------------------------------------------
def v_caso1(t):
    return -g * t

def v_caso2(t):
    return vt * (1.0 - np.exp(-p_caso2 * t))

def v_caso3(t):
    return vt * np.tanh((g / vt_mag) * t)

def metricas(f):
    """RMSE, R^2 y % de barras intersectadas sobre los puntos validos."""
    msk = (t >= T_MIN) & (t <= T_MAX)
    ts, vs, dvs = t[msk], v_bar[msk], dv[msk]
    v_mod = f(ts)
    rmse = np.sqrt(np.mean((v_mod - vs) ** 2))
    ss_res = np.sum((vs - v_mod) ** 2)
    ss_tot = np.sum((vs - np.mean(vs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    pct = 100.0 * np.mean(np.abs(v_mod - vs) <= dvs)
    return rmse, r2, pct

nombre = {
    "caso1": "Caso 1: Caida libre (v = -gt)",
    "caso2": "Caso 2: Arrastre lineal (Stokes)",
    "caso3": "Caso 3: Arrastre cuadratico (Newton)",
}

res = {}
for key, f in [("caso1", v_caso1), ("caso2", v_caso2), ("caso3", v_caso3)]:
    res[key] = metricas(f)

print("\n" + "=" * 78)
print("METRICAS CUANTITATIVAS  (puntos con 0.10 <= t <= 1.17 s)")
print("=" * 78)
print(f"  {'Modelo':<38}{'RMSE [m/s]':>10}{'R^2':>10}{'% barras':>10}")
print("-" * 78)
for key in ("caso1", "caso2", "caso3"):
    rmse, r2, pct = res[key]
    print(f"  {nombre[key]:<38}{rmse:10.4f}{r2:10.4f}{pct:9.1f}%")
print("=" * 78)

# ---------------------------------------------------------------------------
# 6. GRAFICA UNICA: datos experimentales + 3 modelos + velocidad terminal
# ---------------------------------------------------------------------------
tmod = np.linspace(0, T_MAX * 1.15, 300)

plt.figure(figsize=(11, 7))
plt.errorbar(t, v_bar, yerr=dv, fmt="o", color="steelblue", ecolor="steelblue",
             capsize=4, ms=5, zorder=5,
             label="Datos experimentales (vbar +/- dv)")
plt.plot(tmod, v_caso1(tmod), "k--", linewidth=2, label="Caso 1: Caida libre ($v=-gt$)")
plt.plot(tmod, v_caso2(tmod), "b-", linewidth=2, label="Caso 2: Arrastre lineal (Stokes)")
plt.plot(tmod, v_caso3(tmod), "r-", linewidth=2, label="Caso 3: Arrastre cuadratico (Newton)")
plt.axhline(vt, color="green", linestyle=":", linewidth=1.5,
            label=f"$v_t$ exp. = {vt:.2f} m/s")
plt.axhline(0, color="gray", linewidth=0.7)

plt.xlabel("t [s]", fontsize=12)
plt.ylabel("v [m/s]", fontsize=12)
plt.title("Filtro de cafe + 3 clips: Comparacion de modelos vs experimento",
          fontsize=13)
plt.legend(loc="lower left", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG, dpi=300)
print(f"\n  [figura guardada en] {FIG}")

plt.show()

# ---------------------------------------------------------------------------
# 7. RESUMEN / CONCLUSIOn PARA EL REPORTE
# ---------------------------------------------------------------------------
mejor = min(res, key=lambda k: res[k][0])   # menor RMSE
rme, r2, pct = res[mejor]
print("\n" + "=" * 78)
print("CONCLUSION DE LA COMPARACION")
print("=" * 78)
print(f"  Mejor modelo por RMSE: {nombre[mejor]}")
print(f"    RMSE = {rme:.4f} m/s   R^2 = {r2:.4f}   % barras = {pct:.1f}%")
print("  Interpretacion fisica:")
print("    - La caida libre (Caso 1) se descarta: los datos se estabilizan en")
print("      v_t, no crecen linealmente.")
print("    - Los casos 2 y 3 convergen a v_t; el mejor de ambos dependera de")
print("      las metricas anteriores.")
print("    - Criterio (Taylor/Documento): un buen modelo debe intersectar al")
print("      menos 60-70% de las barras de error.")
print("=" * 78)