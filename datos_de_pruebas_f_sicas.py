# -*- coding: utf-8 -*-
"""
Analisis experimental de VELOCIDAD vs TIEMPO  (Teoria de Taylor + Documento
de Analisis Grafico)

Modelo: v(t) = v0 - a*t   -> YA ES LINEAL: no hace falta linealizar.
  Eje Y = v (velocidad)      Eje X = t (tiempo)
  Pendiente m = -a           Ordenada al origen b = v0

Procedimiento:
  1. Tratamiento estadistico: agrupar tiempos practicamente iguales, promedio
     (vbar) y Desviacion Absoluta Maxima (d.a.m.) como incertidumbre dv.
  2. Modelo directamente lineal: v = v0 - a*t
  3. Grafica v vs t con barras de error verticales.
  4. Tres rectas: A (mejor ajuste), B (pend. minima) y C (pend. maxima).
  5. a_exp = -m_A  ;  da = |m_C - m_B|/2  ;  db = |b_C - b_B|/2
  6. Validaciones: aceleracion vs g, velocidad inicial vs 0, intervalo de
     validez (el objeto toca el suelo y v se vuelve >= 0).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. DATOS EXPERIMENTALES (t, v) de las 3 pruebas
# ---------------------------------------------------------------------------
prueba1 = np.array([
    (0.000, -0.33333333), (0.033, -0.31818182), (0.066, -0.17910448),
    (0.100, -0.49253731), (0.133, -0.81818182), (0.166, -0.71641791),
    (0.200, -0.74626866), (0.233, -0.78787879), (0.266, -0.80597015),
    (0.300, -0.83582090), (0.333, -0.95454545), (0.366, -1.07462687),
    (0.400, -1.11940299), (0.433, -1.16666667), (0.466, -1.20895522),
    (0.500, -1.25373134), (0.533, -1.31818182), (0.566, -1.30303030),
    (0.599, -1.34328358), (0.633, -1.38805970), (0.666, -1.33333333),
    (0.699, -1.35820896), (0.733, -1.47761194), (0.766, -1.46969697),
    (0.799, -1.31343284), (0.833, -1.29850746), (0.866, -1.28787879),
    (0.899, -1.22388060), (0.933, -1.23880597), (0.966, -1.31818182),
    (0.999, -1.47761194), (1.033, -1.29850746), (1.066, -1.03030303),
    (1.099, -1.05970149), (1.133, -0.89552239), (1.166, -0.71212121),
    (1.199, -0.95522388), (1.233, -1.16417910), (1.266, -1.13636364),
    (1.299, -1.34328358), (1.333, -1.43283582), (1.366, -1.21212121),
    (1.399, -1.10606061), (1.432, -1.19402985), (1.466, -1.25373134),
    (1.499, -1.10606061), (1.532, -1.05970149), (1.566, -0.98507463),
    (1.599, -1.01515152), (1.632, -1.10447761), (1.666, -1.05970149),
    (1.699, -1.07575758), (1.732, -1.14641791), (1.766, 0.06755196),
])

prueba2 = np.array([
    (0.000, -0.18181818), (0.033, -0.25373134), (0.067, -0.38805970),
    (0.100, -0.51515152), (0.133, -0.58208955), (0.167, -0.67164179),
    (0.200, -0.66666667), (0.233, -0.74626866), (0.267, -0.91044776),
    (0.300, -0.96969697), (0.333, -1.00000000), (0.367, -1.00000000),
    (0.400, -1.09090909), (0.433, -0.92537313), (0.467, -0.88059701),
    (0.500, -1.33333333), (0.533, -1.42424242), (0.566, -1.22388060),
    (0.600, -1.23880597), (0.633, -1.30303030), (0.666, -1.31343284),
    (0.700, -1.31343284), (0.733, -1.37878788), (0.766, -1.38805970),
    (0.800, -1.37313433), (0.833, -1.37878788), (0.866, -1.40298507),
    (0.900, -1.40298507), (0.933, -1.37878788), (0.966, -1.44776119),
    (1.000, -1.35820896), (1.033, -1.27272727), (1.066, -1.32835821),
    (1.100, -1.37313433), (1.133, -1.40909091), (1.166, -1.34328358),
    (1.200, -1.32835821), (1.233, -1.34848485), (1.266, -1.34848485),
    (1.299, -1.32835821), (1.333, -1.32835821), (1.366, -1.36363636),
    (1.399, -1.32835821), (1.433, -1.31343284), (1.466, -1.34848485),
    (1.499, -1.32835821), (1.533, -1.29850746), (1.566, -1.31818182),
    (1.599, -1.26865672), (1.633, -1.22388060), (1.666, -1.29757576),
    (1.699, -0.68313433), (1.733, 0.00955224), (1.766, 0.00936526),
])

prueba3 = np.array([
    (0.000, -0.47058824), (0.034, -0.31343284), (0.067, -0.36363636),
    (0.100, -0.61194030), (0.134, -0.64179104), (0.167, -0.72727273),
    (0.200, -0.70149254), (0.234, -0.70149254), (0.267, -0.98484848),
    (0.300, -1.04477612), (0.334, -1.14925373), (0.367, -1.09090909),
    (0.400, -0.95522388), (0.434, -1.04477612), (0.467, -1.01515152),
    (0.500, -1.31818182), (0.533, -1.43283582), (0.567, -1.32835821),
    (0.600, -1.71212121), (0.633, -1.59701493), (0.667, -1.11940299),
    (0.700, -0.74242424), (0.733, -0.92537313), (0.767, -1.86567164),
    (0.800, -1.77272727), (0.833, -1.04477612), (0.867, -0.97014925),
    (0.900, -1.01515152), (0.933, -1.47761194), (0.967, -1.64179104),
    (1.000, -1.06060606), (1.033, -0.95522388), (1.067, -1.19402985),
    (1.100, -1.24242424), (1.133, -1.10447761), (1.167, -1.11940299),
    (1.200, -1.31818182), (1.233, -1.22388060), (1.267, -1.07462687),
    (1.300, -1.51515152), (1.333, -1.56060606), (1.366, -1.10447761),
    (1.400, -1.10447761), (1.433, -1.28787879), (1.466, -1.37313433),
    (1.500, -1.32835821), (1.533, -1.51515152), (1.566, -1.71641791),
    (1.600, -1.32835821), (1.633, -1.12575758), (1.666, -1.11582090),
    (1.700, -0.74432836), (1.733, -0.60303030), (1.766, 0.02125216),
])

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
# Intervalo de validez: en t > 1.70 s la velocidad se vuelve positiva/cero
# (el objeto choco contra el suelo -> ruido o rebote). Se excluyen esos puntos.
T_MAX_FISICO = 1.70
G_TEORICO    = 9.81    # aceleracion teorica de caida libre (m/s^2)
V0_TEORICO   = 0.0     # velocidad inicial teorica (se suelta del reposo)
# Piso de incertidumbre (si la d.a.m. diera 0)
DV_FLOOR = 0.001

# Guardar las figuras aqui (junto al programa)
CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG1 = os.path.join(CARPETA, "grafica_v_t.png")
FIG2 = os.path.join(CARPETA, "grafica_v_t_validacion.png")

# ---------------------------------------------------------------------------
# 2. TRATAMIENTO ESTADISTICO: promedio + d.a.m. (agrupando tiempos)
# ---------------------------------------------------------------------------
def dam(*valores):
    xbar = np.mean(valores)
    return xbar, np.max(np.abs(valores - xbar))

# Los tiempos de las 3 pruebas coinciden practicamente (diferencia <= 0.001 s),
# asi que se agrupan fila por fila. Se usa el PROMEDIO de los 3 tiempos.
n_filas = min(len(prueba1), len(prueba2), len(prueba3))
filas = []
for i in range(n_filas):
    t_prom = np.mean([prueba1[i, 0], prueba2[i, 0], prueba3[i, 0]])
    vs = np.array([prueba1[i, 1], prueba2[i, 1], prueba3[i, 1]])
    v_bar, dv = dam(vs)
    if dv == 0.0:
        dv = DV_FLOOR
    filas.append((t_prom, v_bar, dv))
filas = np.array(filas, dtype=float)

t     = filas[:, 0]
v_bar = filas[:, 1]
dv    = filas[:, 2]

# ---------------------------------------------------------------------------
# 3. FILTRO POR INTERVALO DE VALIDEZ
# ---------------------------------------------------------------------------
mask = t <= T_MAX_FISICO
tv, vv, dvv = t[mask], v_bar[mask], dv[mask]

print("=" * 78)
print("TABLA DEFINITIVA (v vs t):   t [s]     vbar [m/s]    dv [m/s]")
print("=" * 78)
for i in range(n_filas):
    ok = " (v)" if mask[i] else ""
    print(f"    {t[i]:7.3f}    {v_bar[i]:10.4f}    {dv[i]:9.4f}{ok}")
print(f"\n  Intervalo de validez usado: 0 <= t <= {T_MAX_FISICO} s")
print("  (en t > 1.70 s la velocidad se vuelve >= 0: el objeto choco contra")
print("   el suelo y el modelo v = v0 - a*t deja de describir la realidad)\n")

# Tabla redondeada con la regla de Taylor
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
for i in range(n_filas):
    vr, dvvr = redondear_incertidumbre(v_bar[i], dv[i])
    exp = int(np.floor(np.log10(abs(dvvr))))
    dec = max(0, -exp)
    print(f"  {t[i]:7.3f}   {vr:{8}.{dec}f} +/- {dvvr:.{dec}f}")
print()

# ---------------------------------------------------------------------------
# 4. LAS TRES RECTAS (v = m*t + b)
# ---------------------------------------------------------------------------
# Recta A (mejor ajuste): minimos cuadrados sobre todos los puntos validos.
mA, bA = np.polyfit(tv, vv, 1)

# Recta B (pendiente minima) y Recta C (pendiente maxima): pasan por los
# extremos de la primera y la ultima barra de error validas.
P0, P1 = 0, -1

def recta(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1

# Recta B: superior en la 1a barra, inferior en la ultima
mB, bB = recta(tv[P0], vv[P0] + dvv[P0], tv[P1], vv[P1] - dvv[P1])
# Recta C: inferior en la 1a barra, superior en la ultima
mC, bC = recta(tv[P0], vv[P0] - dvv[P0], tv[P1], vv[P1] + dvv[P1])

# Informacion adicional: recta por los CENTROS de las barras extremas
mCC, bCC = recta(tv[P0], vv[P0], tv[P1], vv[P1])

# Incertidumbres
mExp = mA
dm   = abs(mC - mB) / 2.0
bExp = bA
db   = abs(bC - bB) / 2.0

# Propagar: m = -a  =>  a = -m
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
# 5. VALIDACIONES (Paso 5)
# ---------------------------------------------------------------------------
print("\nVALIDACIONES")
print("-" * 78)

# 5.1 Aceleracion vs g = 9.81
if G_TEORICO - da <= aExp <= G_TEORICO + da:
    print(f"  [OK] g = {G_TEORICO} m/s^2 esta dentro de [{aExp - da:.4f}, {aExp + da:.4f}]")
    print("        -> consistente con caida libre en el vacio.")
else:
    print(f"  [!!] g = {G_TEORICO} m/s^2 NO esta dentro de [{aExp - da:.4f}, {aExp + da:.4f}]")
    print("        -> el experimento NO es caida libre en el vacio; sugiere un")
    print("           plano inclinado o la presencia de friccion/resistencia.")

# 5.2 Velocidad inicial vs 0
if V0_TEORICO - db <= bExp <= V0_TEORICO + db:
    print(f"  [OK] v0 = {V0_TEORICO} m/s esta dentro de [{bExp - db:.4f}, {bExp + db:.4f}]")
    print("        -> consistente con soltar el objeto desde el reposo.")
else:
    print(f"  [!!] v0 = {V0_TEORICO} m/s NO esta dentro de [{bExp - db:.4f}, {bExp + db:.4f}]")
    print("        -> discrepancia significativa en el intercepto, debida al")
    print("           metodo de diferenciacion numerica (dx/dt) usado para")
    print("           calcular la velocidad en el primer instante (error")
    print("           sistematico numerico). En t=0 las velocidades medidas")
    print("           son ~ -0.3, -0.2 y -0.5 m/s, no 0.")
    print(f"           Las rectas B y C anclan en la primera barra (t=0):")
    print(f"           b_B = {bB:.4f}, b_C = {bC:.4f} -> el intercepto de la")
    print("           familia de rectas validas es ~ -0.33 m/s, claramente")
    print("           alejado del 0 teorico.")

# 5.3 Intervalo de validez
print(f"\n  Intervalo de validez del modelo: 0 <= t <= {T_MAX_FISICO} s")
print(f"  Para t > {T_MAX_FISICO} s el objeto ya toco el suelo: la velocidad se")
print("  vuelve positiva o nula y el modelo v = v0 - a*t deja de ser valido.\n")

# ---------------------------------------------------------------------------
# 6. GRAFICA 1: v vs t con barras de error y las tres rectas
# ---------------------------------------------------------------------------
tg = np.linspace(0, 1.9, 100)

plt.figure(figsize=(10, 7))
plt.errorbar(t, v_bar, yerr=dv, fmt="o", color="black", ecolor="red",
             capsize=4, ms=5, label="Datos (vbar +/- dv)")
plt.plot(tg, mA * tg + bA, "b-", linewidth=2.2,
         label=f"Recta A (mejor): m = {mA:.4f}")
plt.plot(tg, mB * tg + bB, "g--", linewidth=1.5,
         label=f"Recta B (min):   m = {mB:.4f}")
plt.plot(tg, mC * tg + bC, "m--", linewidth=1.5,
         label=f"Recta C (max):   m = {mC:.4f}")
plt.axvspan(T_MAX_FISICO, tg.max(), color="gray", alpha=0.15,
            label=f"Fuera de validez (t > {T_MAX_FISICO} s)")
plt.axhline(0, color="k", linewidth=0.8)

plt.xlabel("t  [s]")
plt.ylabel("v  [m/s]")
plt.title("Velocidad vs tiempo con barras de error (las tres rectas)")
plt.legend(loc="best", fontsize=9)
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
plt.title("Velocidad vs tiempo: datos y modelo ajustado")
plt.legend(loc="best", fontsize=9)
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
print("  1. Tabla t, vbar, dv (d.a.m.)  -> impresa arriba")
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
print("  Conclusion: la velocidad es derivada numerica de x (dx/dt), por lo")
print("  que amplifica el ruido (barras de error grandes). La pendiente sigue")
print("  siendo la aceleracion, pero el intercepto NO es 0 por el error")
print("  sistematico del metodo de diferenciacion en t = 0.")
print("=" * 78)