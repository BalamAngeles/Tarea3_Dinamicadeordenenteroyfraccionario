# -*- coding: utf-8 -*-
"""
Analisis experimental de VELOCIDAD vs TIEMPO (masa_A - Pruebas 2.1, 2.2, 2.3)
Teoria de Taylor + Documento de Analisis Grafico.

Modelo: v(t) = v0 - a*t  -> YA ES LINEAL.
  Eje Y = v (velocidad)    Eje X = t (tiempo)
  Pendiente m = -a         Ordenada al origen b = v0

Procedimiento:
  1. Tratamiento estadistico: los tiempos de las 3 pruebas NO coinciden fila a
     fila, asi que se agrupan por TOLERANCIA (tiempos practicamente iguales).
     Se calcula el promedio (vbar) y la d.a.m. como incertidumbre dv.
  2. Modelo directamente lineal: v = v0 - a*t
  3. Grafica v vs t con barras de error verticales.
  4. Tres rectas: A (mejor ajuste), B (pend. minima) y C (pend. maxima).
  5. a_exp = -m_A  ;  da = |m_C - m_B|/2  ;  db = |b_C - b_B|/2
  6. Validaciones y intervalo de validez (v se vuelve >= 0 al tocar el suelo).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. DATOS EXPERIMENTALES (t, v) de las 3 pruebas (masa_A)
# ---------------------------------------------------------------------------
prueba1 = np.array([
    (0.000, -0.11764706), (0.034, -0.17910448), (0.067, -0.25757576),
    (0.100, -0.47761194), (0.134, -0.67164179), (0.167, -0.72727273),
    (0.200, -0.87878788), (0.233, -1.20895522), (0.267, -1.25373134),
    (0.300, -1.00000000), (0.333, -1.17910448), (0.367, -1.46268657),
    (0.400, -1.50000000), (0.433, -1.53731343), (0.467, -1.59701493),
    (0.500, -1.65151515), (0.533, -1.67164179), (0.567, -1.70149254),
    (0.600, -1.77272727), (0.633, -1.74626866), (0.667, -1.74626866),
    (0.700, -1.63636364), (0.733, -1.62686567), (0.767, -1.80597015),
    (0.800, -1.83333333), (0.833, -1.77611940), (0.867, -1.77611940),
    (0.900, -1.83333333), (0.933, -1.80597015), (0.967, -1.79104478),
    (1.000, -1.80303030), (1.033, -1.78787879), (1.066, -1.50746269),
    (1.100, -1.65671642), (1.133, -1.95454545), (1.166, -1.76119403),
    (1.200, -1.73134328), (1.233, -1.77272727), (1.266, 0.18329278),
])

prueba2 = np.array([
    (0.000, -0.72727273), (0.033, -0.81818182), (0.066, -1.56716418),
    (0.100, -1.77611940), (0.133, -1.31818182), (0.166, -1.07462687),
    (0.200, -1.05970149), (0.233, -1.42000000), (0.300, -1.72000000),
    (0.333, -1.95454545), (0.366, -1.70149254), (0.400, -1.68656716),
    (0.433, -1.51515152), (0.466, -1.67164179), (0.500, -1.88059701),
    (0.533, -1.62121212), (0.566, -1.59701493), (0.600, -1.71641791),
    (0.633, -1.74242424), (0.666, -1.76119403), (0.700, -1.64179104),
    (0.733, -1.51515152), (0.766, -1.63636364), (0.799, -1.70149254),
    (0.833, -1.67164179), (0.866, -1.71212121), (0.899, -1.70149254),
    (0.933, -1.71641791), (0.966, -1.72727273), (0.999, -1.70149254),
    (1.033, -1.74626866), (1.066, -1.77272727), (1.099, -1.74626866),
    (1.133, -1.76119403), (1.166, -1.77939394), (1.199, -1.72716418),
    (1.233, -0.99910448), (1.266, 0.00995945),
])

prueba3 = np.array([
    (0.000, -0.57575758), (0.033, -0.56060606), (0.066, -0.62686567),
    (0.100, -0.73134328), (0.133, -0.83333333), (0.166, -1.13432836),
    (0.200, -1.05970149), (0.233, -1.07575758), (0.266, -1.43283582),
    (0.300, -1.50746269), (0.333, -1.53030303), (0.366, -1.55223881),
    (0.400, -1.59701493), (0.433, -1.84848485), (0.466, -1.73134328),
    (0.500, -1.55223881), (0.533, -1.31818182), (0.566, -1.35820896),
    (0.600, -1.76119403), (0.633, -1.74242424), (0.666, -1.68656716),
    (0.700, -1.71641791), (0.733, -1.81818182), (0.766, -2.27272727),
    (0.799, -2.22388060), (0.833, -1.79104478), (0.866, -1.81818182),
    (0.899, -1.38805970), (0.933, -1.32835821), (0.966, -1.71212121),
    (0.999, -1.71641791), (1.033, -1.74626866), (1.066, -1.77272727),
    (1.099, -1.74626866), (1.133, -1.79104478), (1.166, -1.80303030),
    (1.199, -1.76119403), (1.233, -1.71597015), (1.266, 0.10300081),
])

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
# Los tiempos de cada prueba difieren levemente (0.033 vs 0.034, etc.): un dato
# se agrupa con el anterior si su tiempo difiere en menos de esta tolerancia.
TOL_AGRUPACION = 0.01   # segundos

# Intervalo de validez: en t > 1.233 s la velocidad se vuelve positiva/cero
# (el objeto toco el suelo -> ruido/rebote). Se excluyen esos puntos.
T_MAX_FISICO = 1.24
G_TEORICO    = 9.81     # aceleracion teorica de caida libre (m/s^2)
V0_TEORICO   = 0.0      # velocidad inicial teorica (reposo)
DV_FLOOR     = 0.001    # piso de incertidumbre si la d.a.m. diera 0

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG1 = os.path.join(CARPETA, "2grafica_v_t.png")
FIG2 = os.path.join(CARPETA, "2grafica_v_t_validacion.png")

# ---------------------------------------------------------------------------
# 2. TRATAMIENTO ESTADISTICO: agrupar por tolerancia + promedio + d.a.m.
# ---------------------------------------------------------------------------
def dam(valores):
    xbar = np.mean(valores)
    return xbar, np.max(np.abs(valores - xbar))

# Juntar todos los pares (t, v) y ordenarlos por tiempo
pares = np.concatenate([prueba1, prueba2, prueba3])
pares = pares[np.argsort(pares[:, 0])]

# Agrupar tiempos practicamente iguales
grupos = []              # cada grupo: lista de tiempos y lista de velocidades
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
    ok = " (v)" if t[i] <= T_MAX_FISICO else ""
    print(f"    {t[i]:7.3f}    {v_bar[i]:10.4f}    {dv[i]:9.4f}   {len(ts)}{ok}")
print(f"\n  Intervalo de validez usado: 0 <= t <= {T_MAX_FISICO} s")
print("  (en t > 1.233 s la velocidad se vuelve >= 0: el objeto toca el suelo")
print("   y el modelo v = v0 - a*t deja de describir la realidad)\n")

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
# 3. FILTRO POR INTERVALO DE VALIDEZ
# ---------------------------------------------------------------------------
mask = t <= T_MAX_FISICO
tv, vv, dvv = t[mask], v_bar[mask], dv[mask]

# ---------------------------------------------------------------------------
# 4. LAS TRES RECTAS (v = m*t + b)
# ---------------------------------------------------------------------------
# Recta A (mejor ajuste): minimos cuadrados sobre todos los puntos validos.
mA, bA = np.polyfit(tv, vv, 1)

P0, P1 = 0, -1

def recta(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1

# Recta B (pend. minima): superior en 1a barra, inferior en la ultima
mB, bB = recta(tv[P0], vv[P0] + dvv[P0], tv[P1], vv[P1] - dvv[P1])
# Recta C (pend. maxima): inferior en 1a barra, superior en la ultima
mC, bC = recta(tv[P0], vv[P0] - dvv[P0], tv[P1], vv[P1] + dvv[P1])

# Informacion adicional: recta por los CENTROS de las barras extremas
mCC, bCC = recta(tv[P0], vv[P0], tv[P1], vv[P1])

mExp = mA
dm   = abs(mC - mB) / 2.0
bExp = bA
db   = abs(bC - bB) / 2.0

aExp = -mExp
da   = dm

print("=" * 78)
print("RESULTADOS DE LAS TRES RECTAS  (grafica: v vs t)")
print("=" * 78)
print(f"  Recta A (mejor ajuste, min. cuadrados): m_A = {mA:9.4f}   b_A = {bA:8.4f}")
print(f"  Recta B (pendiente minima):             m_B = {mB:9.4f}   b_B = {bB:8.4f}")
print(f"  Recta C (pendiente maxima):             m_C = {mC:9.4f}   b_C = {bC:8.4f}")
print(f"  [referencia] centro de barras extremas: m   = {mCC:9.4f}   b   = {bCC:8.4f}")
print("-" * 78)
print(f"  m_exp = {mExp:.4f} +/- {dm:.4f}   (dm = |m_C - m_B|/2)")
print(f"  b_exp = {bExp:.4f} +/- {db:.4f} m/s  (db = |b_C - b_B|/2)")
print(f"  a_exp = -m_exp = {aExp:.4f} +/- {da:.4f} m/s^2")
print("=" * 78)

# ---------------------------------------------------------------------------
# 5. VALIDACIONES
# ---------------------------------------------------------------------------
print("\nVALIDACIONES")
print("-" * 78)

if G_TEORICO - da <= aExp <= G_TEORICO + da:
    print(f"  [OK] g = {G_TEORICO} m/s^2 esta dentro de [{aExp - da:.4f}, {aExp + da:.4f}]")
    print("        -> consistente con caida libre en el vacio.")
else:
    print(f"  [!!] g = {G_TEORICO} m/s^2 NO esta dentro de [{aExp - da:.4f}, {aExp + da:.4f}]")
    print("        -> el experimento NO es caida libre en el vacio; sugiere un")
    print("           plano inclinado o la presencia de friccion/resistencia.")

if V0_TEORICO - db <= bExp <= V0_TEORICO + db:
    print(f"  [OK] v0 = {V0_TEORICO} m/s esta dentro de [{bExp - db:.4f}, {bExp + db:.4f}]")
    print("        -> consistente con soltar el objeto desde el reposo.")
else:
    print(f"  [!!] v0 = {V0_TEORICO} m/s NO esta dentro de [{bExp - db:.4f}, {bExp + db:.4f}]")
    print("        -> discrepancia significativa en el intercepto, debida al")
    print("           metodo de diferenciacion numerica (dx/dt) usado para")
    print("           calcular la velocidad en el primer instante (error")
    print("           sistematico numerico). En t=0 las velocidades medidas")
    print("           son ~ -0.1, -0.7 y -0.6 m/s, no 0.")
    print(f"           Las rectas B y C anclan en la primera barra (t=0):")
    print(f"           b_B = {bB:.4f}, b_C = {bC:.4f} -> el intercepto de la")
    print("           familia de rectas validas queda claramente alejado del 0.")

print(f"\n  Intervalo de validez del modelo: 0 <= t <= {T_MAX_FISICO} s")
print(f"  Para t > {T_MAX_FISICO} s el objeto ya toco el suelo: la velocidad se")
print("  vuelve positiva o nula y el modelo v = v0 - a*t deja de ser valido.\n")

# ---------------------------------------------------------------------------
# 6. GRAFICA 1: v vs t con barras de error y las tres rectas
# ---------------------------------------------------------------------------
xmin, xmax = 0, (tv[P1] * 1.15)
tg = np.linspace(xmin, xmax, 100)

plt.figure(figsize=(10, 7))
plt.errorbar(t, v_bar, yerr=dv, fmt="o", color="black", ecolor="red",
             capsize=4, ms=5, label="Datos (vbar +/- dv)")
plt.plot(tg, mA * tg + bA, "b-", linewidth=2.2,
         label=f"Recta A (mejor): m = {mA:.4f}")
plt.plot(tg, mB * tg + bB, "g--", linewidth=1.5,
         label=f"Recta B (min):   m = {mB:.4f}")
plt.plot(tg, mC * tg + bC, "m--", linewidth=1.5,
         label=f"Recta C (max):   m = {mC:.4f}")
plt.axvspan(T_MAX_FISICO, xmax, color="gray", alpha=0.15,
            label=f"Fuera de validez (t > {T_MAX_FISICO} s)")
plt.axhline(0, color="k", linewidth=0.8)

plt.xlabel("t  [s]")
plt.ylabel("v  [m/s]")
plt.title("Velocidad vs tiempo - masa_A (las tres rectas)")
plt.legend(loc="lower left", fontsize=9)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(FIG1, dpi=150)
print(f"  [figura 1 guardada en] {FIG1}")

# ---------------------------------------------------------------------------
# 7. GRAFICA 2: modelo ajustado y puntos validos/no validos
# ---------------------------------------------------------------------------
tmod = np.linspace(0, tv[P1] * 1.05, 200)
vmod = mExp * tmod + bExp

plt.figure(figsize=(10, 6))
plt.errorbar(t[mask], v_bar[mask], yerr=dv[mask], fmt="o", color="black",
             ecolor="red", capsize=4, ms=5,
             label=f"Datos validos (0 <= t <= {T_MAX_FISICO} s)")
plt.errorbar(t[~mask], v_bar[~mask], yerr=dv[~mask], fmt="s", color="gray",
             ecolor="gray", capsize=4, ms=5, label="Datos fuera de validez")
plt.plot(tmod, vmod, "b-", linewidth=2,
         label=f"Modelo: v = {bExp:.3f} + ({mExp:.3f}) t")
plt.axvline(T_MAX_FISICO, color="orange", linestyle=":", linewidth=1.5,
            label=f"Fin de validez (t = {T_MAX_FISICO} s)")
plt.axhline(0, color="k", linewidth=0.8)

plt.xlabel("t  [s]")
plt.ylabel("v  [m/s]")
plt.title("Velocidad vs tiempo - masa_A: datos y modelo ajustado")
plt.legend(loc="lower left", fontsize=9)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(FIG2, dpi=150)
print(f"  [figura 2 guardada en] {FIG2}")

plt.show()

# ---------------------------------------------------------------------------
# 8. RESUMEN PARA EL REPORTE
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("RESUMEN PARA EL REPORTE")
print("=" * 78)
print("  1. Tabla t, vbar, dv (d.a.m.)  -> impresa arriba (agrupado por")
print("     tolerancia, porque los tiempos de las 3 pruebas no coinciden)")
print("  2. Grafica v vs t con barras de error  -> Ventana 1 / figura 1")
print("  3. Tres rectas A (mejor), B (min) y C (max) -> Ventana 1")
print(f"  4. a_exp = {aExp:.4f} +/- {da:.4f} m/s^2")
print(f"     v0_exp = {bExp:.4f} +/- {db:.4f} m/s   (teorico: {V0_TEORICO} m/s)")
print(f"  5. Intervalo de validez: 0 <= t <= {T_MAX_FISICO} s")
print(f"     g = {G_TEORICO} m/s^2 "
      + ("SI" if G_TEORICO - da <= aExp <= G_TEORICO + da else "NO")
      + f" cae en [{aExp - da:.2f}, {aExp + da:.2f}]")
print(f"     v0 = {V0_TEORICO} m/s "
      + ("SI" if V0_TEORICO - db <= bExp <= V0_TEORICO + db else "NO")
      + f" cae en [{bExp - db:.2f}, {bExp + db:.2f}]")
print("  Conclusion: v es la derivada numerica de x (dx/dt), lo que amplifica")
print("  el ruido. La pendiente da la aceleracion, pero el intercepto no es 0")
print("  por el error sistematico de la diferenciacion en t = 0.")
print("=" * 78)