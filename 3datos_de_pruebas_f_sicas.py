# -*- coding: utf-8 -*-
"""
Analisis experimental de VELOCIDAD vs TIEMPO (Pruebas 3.1, 3.2, 3.3)
Teoria de Taylor + Documento de Analisis Grafico.
Version CORREGIDA para resolver los problemas de congruencia:

  [C1] Se EXCLUYEN los primeros puntos ruidosos (t < T_MIN_FISICO), donde la
       velocidad en t=0 es muy variable (-0.67, -0.06, -0.56 m/s) por el metodo
       de diferenciacion numerica.
  [C2] La Recta C (pendiente maxima = menos inclinada) se RESTRINGE a pendiente
       negativa o cero (fisicamente imposible que la velocidad aumente en caida).
  [C3] Se reporta como resultado PRINCIPAL el ajuste por minimos cuadrados
       (scipy.stats.linregress) CON su error estandar, ademas del metodo
       grafico de las tres rectas (con la correccion C2).

Modelo: v(t) = v0 - a*t  -> YA ES LINEAL.
  Pendiente m = -a   Ordenada al origen b = v0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# 1. DATOS EXPERIMENTALES (t, v) de las 3 pruebas
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
# CONFIGURACION
# ---------------------------------------------------------------------------
TOL_AGRUPACION = 0.01     # segundos (agrupado de tiempos)
T_MIN_FISICO   = 0.10     # CORRECCION C1: excluir t < 0.1 s (puntos iniciales
                          #   ruidosos por la diferenciacion numerica)
T_MAX_FISICO   = 1.17     # validez: en t > 1.166 s la v se vuelve >= 0
G_TEORICO      = 9.81
V0_TEORICO     = 0.0
DV_FLOOR       = 0.001

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG1 = os.path.join(CARPETA, "3grafica_v_t.png")
FIG2 = os.path.join(CARPETA, "3grafica_v_t_validacion.png")

# ---------------------------------------------------------------------------
# 2. TRATAMIENTO ESTADISTICO: agrupar por tolerancia + promedio + d.a.m.
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

filas = []
for ts, vs in grupos:
    t_prom = np.mean(ts)
    v_bar, dv = dam(np.array(vs))
    if dv == 0.0:
        dv = DV_FLOOR
    filas.append((t_prom, v_bar, dv))
filas = np.array(filas, dtype=float)

t     = filas[:, 0]
v_bar = filas[:, 1]
dv    = filas[:, 2]

print("=" * 78)
print("RESUMEN DEL AGRUPADO POR TOLERANCIA")
print("=" * 78)
print(f"  Total de pares (t, v) de las 3 pruebas : {len(pares)}")
print(f"  Grupos formados (tiempos iguales)      : {len(filas)}")
print(f"  Tolerancia usada                      : {TOL_AGRUPACION} s")
print()
print("=" * 78)
print("TABLA DEFINITIVA (v vs t):   t [s]      vbar [m/s]    dv [m/s]  (n)")
print("=" * 78)
for i, (ts, vs) in enumerate(grupos):
    if t[i] < T_MIN_FISICO:
        etiq = " [excl]"        # excluido por ruido inicial (C1)
    elif t[i] <= T_MAX_FISICO:
        etiq = " (v)"           # punto valido usado en el ajuste
    else:
        etiq = " [excl]"        # excluido por fin de validez
    print(f"    {t[i]:7.3f}    {v_bar[i]:10.4f}    {dv[i]:9.4f}   {len(ts)}{etiq}")
print(f"\n  Correccion C1: ajuste solo con {T_MIN_FISICO} s <= t <= {T_MAX_FISICO} s")
print("  (se excluyen los primeros puntos ruidosos y los de pos-impacto)\n")

def redondear_incertidumbre(val, inc):
    if inc <= 0:
        return val, 0.0
    exp = int(np.floor(np.log10(abs(inc))))
    dxr = np.round(inc / (10 ** exp)) * 10 ** exp
    vr = np.round(val / (10 ** exp)) * 10 ** exp
    return vr, dxr

print("  TABLA REDONDEADA (regla de Taylor: dv a 1 cifra significativa)")
print("  " + "-" * 60)
print(f"  {'t [s]':>7}   {'vbar +/- dv [m/s]':>20}")
print("  " + "-" * 60)
for i in range(len(filas)):
    vr, dvvr = redondear_incertidumbre(v_bar[i], dv[i])
    exp = int(np.floor(np.log10(abs(dvvr))))
    dec = max(0, -exp)
    print(f"  {t[i]:7.3f}   {vr:{8}.{dec}f} +/- {dvvr:.{dec}f}")
print()

# ---------------------------------------------------------------------------
# 3. MASCARA DE AJUSTE (excluye primeros puntos ruidosos y pos-impacto)
# ---------------------------------------------------------------------------
mask_fit = (t >= T_MIN_FISICO) & (t <= T_MAX_FISICO)
tv, vv, dvv = t[mask_fit], v_bar[mask_fit], dv[mask_fit]

# ---------------------------------------------------------------------------
# 4a. RESULTADO PRINCIPAL (CORRECCION C3): MINIMOS CUADRADOS con error estandar
# ---------------------------------------------------------------------------
mLS, bLS, r, p, s_m = stats.linregress(tv, vv)
dmLS = s_m                       # error estandar de la pendiente
dbLS = s_m * np.sqrt(np.mean(tv ** 2))  # error estandar del intercepto
R2   = r ** 2

# ---------------------------------------------------------------------------
# 4b. METODO GRAFICO DE LAS TRES RECTAS (corregido: C2)
# ---------------------------------------------------------------------------
# Primera y ultima barra dentro del rango de ajuste (C1). Los puntos iniciales
# ruidosos ya no participan.
P0, P1 = 0, -1

def recta(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1

# Recta B (m_min, la mas inclinada hacia abajo): top de la 1a barra, fondo ult.
mB, bB = recta(tv[P0], vv[P0] + dvv[P0], tv[P1], vv[P1] - dvv[P1])
# Recta C (m_max, la menos inclinada): fondo de la 1a barra, top de la ultima.
mC, bC = recta(tv[P0], vv[P0] - dvv[P0], tv[P1], vv[P1] + dvv[P1])

# CORRECCION C2: fisicamente m no puede ser positiva en caida (la velocidad
# no aumenta). Si la recta C saliera con pendiente positiva, se limita a 0
# (la "menos inclinada posible" es una horizontal).
if mC > 0.0:
    print(f"  [Correccion C2] m_C calculada = {mC:.4f} (>0, fisicamente imposible)")
    print(f"  -> se limita a m_C = 0 (recta horizontal, menos inclinada posible).")
    mC = 0.0
    bC = vv[P0] - dvv[P0] - mC * tv[P0]   # pasa por el fondo de la 1a barra

# (por construccion mB <= mC, ya que mB usa el punto mas alto y el mas bajo)
mB = min(mB, mC)

# Recta A grafica (correccion de congruencia): por el CENTRO de la 1a y ultima
# barra del rango de ajuste (metodo del documento).
mA, bA = recta(tv[P0], vv[P0], tv[P1], vv[P1])

mExp = mLS            # valor del ajuste por minimos cuadrados
dm   = abs(mC - mB) / 2.0
bExp = bLS
db   = abs(bC - bB) / 2.0

aLS   = -mLS
daLS  = dmLS
aExp  = -mA           # aceleracion por el metodo grafico
da    = abs(mC - mB) / 2.0

print("=" * 78)
print("RESULTADO PRINCIPAL - MINIMOS CUADRADOS (correccion C3)")
print("=" * 78)
print(f"  m_LS = {mLS:.4f} +/- {dmLS:.4f}  m/s^2  (error estandar de scipy)")
print(f"  b_LS = {bLS:.4f} m/s   (intercepto)")
print(f"  R^2  = {R2:.4f}")
print(f"  a_exp = -m_LS = {aLS:.4f} +/- {daLS:.4f} m/s^2")
print("-" * 78)
print("METODO GRAFICO DE LAS TRES RECTAS (corregido)")
print("-" * 78)
print(f"  Recta A (por centros de 1a y ultima barra): m_A = {mA:9.4f}   b_A = {bA:8.4f}")
print(f"  Recta B (pendiente minima / mas inclinada): m_B = {mB:9.4f}   b_B = {bB:8.4f}")
print(f"  Recta C (pendiente maxima / menos inclinada):m_C = {mC:9.4f}   b_C = {bC:8.4f}")
print(f"  a_exp (grafico) = {aExp:.4f} +/- {da:.4f} m/s^2")
print("=" * 78)

# ---------------------------------------------------------------------------
# 5. VALIDACIONES
# ---------------------------------------------------------------------------
print("\nVALIDACIONES")
print("-" * 78)

# 5.1 Aceleracion vs g (usando el resultado por minimos cuadrados)
if G_TEORICO - daLS <= aLS <= G_TEORICO + daLS:
    print(f"  [OK] g = {G_TEORICO} m/s^2 esta dentro de [{aLS - daLS:.4f}, {aLS + daLS:.4f}]")
    print("        -> consistente con caida libre en el vacio.")
else:
    print(f"  [!!] g = {G_TEORICO} m/s^2 NO esta dentro de [{aLS - daLS:.4f}, {aLS + daLS:.4f}]")
    print("        -> el experimento NO es caida libre en el vacio; sugiere un")
    print("           plano inclinado o la presencia de friccion/resistencia.")

# 5.2 Velocidad inicial vs 0 (intercepto del ajuste por minimos cuadrados)
if V0_TEORICO - dbLS <= bLS <= V0_TEORICO + dbLS:
    print(f"  [OK] v0 = {V0_TEORICO} m/s esta dentro de [{bLS - dbLS:.4f}, {bLS + dbLS:.4f}]")
    print("        -> consistente con soltar el objeto desde el reposo.")
else:
    print(f"  [!!] v0 = {V0_TEORICO} m/s NO esta dentro de [{bLS - dbLS:.4f}, {bLS + dbLS:.4f}]")
    print("        -> discrepancia significativa en el intercepto, debida al")
    print("           metodo de diferenciacion numerica (dx/dt) por t=0.")
    print("           (por eso se excluyen los puntos t < 0.1 s en el ajuste)")

print(f"\n  Intervalo de validez del modelo: {T_MIN_FISICO} <= t <= {T_MAX_FISICO} s")
print(f"  Para t > {T_MAX_FISICO} s el objeto ya toco el suelo: la velocidad se")
print("  vuelve positiva o nula y el modelo v = v0 - a*t deja de ser valido.\n")

# ---------------------------------------------------------------------------
# 6. GRAFICA 1: v vs t con puntos validez e iniciales excluidos + las 3 rectas
# ---------------------------------------------------------------------------
xmin, xmax = 0, tv[P1] * 1.15
tg = np.linspace(xmin, xmax, 100)

plt.figure(figsize=(10, 7))
# Puntos iniciales excluidos (t < T_MIN)
m0 = ~mask_fit & (t < T_MIN_FISICO)
plt.errorbar(t[m0], v_bar[m0], yerr=dv[m0], fmt="s", color="gray",
             ecolor="gray", capsize=4, ms=5, label="Puntos iniciales excluidos (t < 0.1 s)")
plt.errorbar(tv, vv, yerr=dvv, fmt="o", color="black", ecolor="red",
             capsize=4, ms=5, label="Puntos del ajuste (vbar +/- dv)")
plt.plot(tg, mLS * tg + bLS, "b-", linewidth=2.2,
         label=f"Recta A (min. cuadrados): m = {mLS:.4f}")
plt.plot(tg, mA * tg + bA, "c--", linewidth=1.2,
         label=f"Recta A' (centros extremos): m = {mA:.4f}")
plt.plot(tg, mB * tg + bB, "g--", linewidth=1.5,
         label=f"Recta B (min):   m = {mB:.4f}")
plt.plot(tg, mC * tg + bC, "m--", linewidth=1.5,
         label=f"Recta C (max):   m = {mC:.4f}")
plt.axvspan(T_MAX_FISICO, xmax, color="gray", alpha=0.15,
            label=f"Fuera de validez (t > {T_MAX_FISICO} s)")
plt.axhline(0, color="k", linewidth=0.8)

plt.xlabel("t  [s]")
plt.ylabel("v  [m/s]")
plt.title("Velocidad vs tiempo - Pruebas 3.x (corregido: sin t<0.1, m_C<=0)")
plt.legend(loc="lower left", fontsize=8)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(FIG1, dpi=150)
print(f"  [figura 1 guardada en] {FIG1}")

# ---------------------------------------------------------------------------
# 7. GRAFICA 2: modelo ajustado y puntos validos/no validos
# ---------------------------------------------------------------------------
tmod = np.linspace(0, tv[P1] * 1.05, 200)
vmod = mLS * tmod + bLS

plt.figure(figsize=(10, 6))
plt.errorbar(t[m0], v_bar[m0], yerr=dv[m0], fmt="s", color="gray",
             ecolor="gray", capsize=4, ms=5, label="Puntos iniciales excluidos")
plt.errorbar(tv, vv, yerr=dvv, fmt="o", color="black", ecolor="red",
             capsize=4, ms=5, label=f"Datos del ajuste (t >= {T_MIN_FISICO} s)")
plt.errorbar(t[~mask_fit & (t > T_MAX_FISICO)], v_bar[~mask_fit & (t > T_MAX_FISICO)],
             yerr=dv[~mask_fit & (t > T_MAX_FISICO)], fmt="s", color="darkgray",
             ecolor="darkgray", capsize=4, ms=5, label="Datos fuera de validez")
plt.plot(tmod, vmod, "b-", linewidth=2,
         label=f"Modelo: v = {bLS:.3f} + ({mLS:.3f}) t  (R^2 = {R2:.3f})")
plt.axvline(T_MAX_FISICO, color="orange", linestyle=":", linewidth=1.5,
            label=f"Fin de validez (t = {T_MAX_FISICO} s)")
plt.axvline(T_MIN_FISICO, color="purple", linestyle=":", linewidth=1.5,
            label=f"Inicio de ajuste (t = {T_MIN_FISICO} s)")
plt.axhline(0, color="k", linewidth=0.8)

plt.xlabel("t  [s]")
plt.ylabel("v  [m/s]")
plt.title("Velocidad vs tiempo - Pruebas 3.x: modelo ajustado (corregido)")
plt.legend(loc="lower left", fontsize=8)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(FIG2, dpi=150)
print(f"  [figura 2 guardada en] {FIG2}")

plt.show()

# ---------------------------------------------------------------------------
# 8. RESUMEN PARA EL REPORTE (CON RESULTADOS CORREGIDOS)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("RESUMEN PARA EL REPORTE  (CONGRUENCIA CORREGIDA)")
print("=" * 78)
print(f"  Ajuste: minimos cuadrados, t en [{T_MIN_FISICO}, {T_MAX_FISICO}] s")
print(f"  a_exp = {aLS:.4f} +/- {daLS:.4f} m/s^2   (R^2 = {R2:.4f})")
print(f"  v0_exp = {bLS:.4f} +/- {dbLS:.4f} m/s  (teorico: {V0_TEORICO} m/s)")
print(f"  Metodo grafico (3 rectas): a = {aExp:.4f} +/- {da:.4f} m/s^2")
print("  Correcciones aplicadas:")
print(f"   C1. Excluidos puntos con t < {T_MIN_FISICO} s (ruido inicial)")
print(f"   C2. Recta C limitada a m <= 0 (velocidad no aumenta en caida)")
print("   C3. Resultado principal por minimos cuadrados (scipy) con su error")
print(f"  g = {G_TEORICO} m/s^2 "
      + ("SI" if G_TEORICO - daLS <= aLS <= G_TEORICO + daLS else "NO")
      + f" cae en [{aLS - daLS:.2f}, {aLS + daLS:.2f}]")
print(f"  v0 = {V0_TEORICO} m/s "
      + ("SI" if V0_TEORICO - dbLS <= bLS <= V0_TEORICO + dbLS else "NO")
      + f" cae en [{bLS - dbLS:.2f}, {bLS + dbLS:.2f}]")
print("  Conclusion: v es la derivada numerica de x (dx/dt), lo que amplifica")
print("  el ruido. Excluir los primeros puntos reduce el sesgo del intercepto")
print("  y la correccion C2 evita pendientes fisicamente imposibles.")
print("=" * 78)