"""
================================================================================
 ANALISIS EXPERIMENTAL - CAIDA DE UN FILTRO DE CAFE CON 3 CLIPS
 Apartado 5.1 (Analisis experimental) - Ajuste por MINIMOS CUADRADOS
 Curso: Temas selectos de matematicas / dinamica de sistemas de orden entero
        y fraccionario - UNAM

 Sigue la misma logica que los casos "sin clips" y "con 2 clips" del reporte
 (promedio de v(t) entre repeticiones + barra de error = Desviacion Absoluta
 Maxima, d.a.m.), pero en vez de las "rectas limite" (metodo grafico manual
 A/B/C) se ajusta v(t) = a*t + b por MINIMOS CUADRADOS. La incertidumbre de
 la pendiente/ordenada usa las formulas estandar de regresion lineal
 (Taylor, "An Introduction to Error Analysis", cap. 8-9).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# 1. DATOS EXPERIMENTALES (filtro de cafe + 3 clips, 3 repeticiones)
# ------------------------------------------------------------------------
t1 = np.array([0,0.033,0.066,0.1,0.133,0.166,0.2,0.233,0.266,0.3,0.333,0.366,
               0.4,0.433,0.466,0.5,0.533,0.566,0.6,0.633,0.666,0.7,0.733,
               0.766,0.8,0.833,0.866,0.899,0.933,0.966,0.999,1.033,1.066,
               1.099,1.133,1.166,1.199])
v1 = np.array([-0.66666667,-0.77272727,-0.92537313,-1.04477612,-1.16666667,
               -1.05970149,-1.23880597,-1.45454545,-1.35820896,-1.47761194,
               -1.68181818,-1.7761194,-1.80597015,-1.86363636,-1.85074627,
               -1.8358209,-1.8030303,-1.76119403,-1.85074627,-1.98484848,
               -1.97014925,-1.97014925,-2.06060606,-2.07462687,-1.91044776,
               -1.90909091,-2.06060606,-2.05970149,-2.02985075,-2.0,
               -1.95522388,-1.95522388,-1.96969697,-1.89686567,-1.4338806,
               -0.41469697,0.03510292])

t2 = np.array([0,0.033,0.066,0.1,0.133,0.166,0.2,0.233,0.266,0.3,0.333,0.366,
               0.4,0.433,0.466,0.5,0.533,0.566,0.599,0.633,0.666,0.699,0.733,
               0.766,0.799,0.833,0.866,0.899,0.933,0.966,0.999,1.033,1.066,
               1.099,1.133,1.166,1.199])
v2 = np.array([-0.06060606,-0.45454545,-0.95522388,-1.01492537,-0.92424242,
               -1.01492537,-1.08955224,-1.10606061,-1.1641791,-1.3880597,
               -1.51515152,-1.52238806,-2.11940299,-2.18181818,-1.8358209,
               -1.94029851,-2.0,-1.77272727,-2.02985075,-2.10447761,
               -1.8030303,-1.86567164,-1.74626866,-1.77272727,-2.53731343,
               -2.62686567,-2.03030303,-2.0,-1.98507463,-2.01515152,-2.0,
               -1.98507463,-2.01515152,-1.97014925,-1.95447761,-1.4619697,
               0.01548027])

t3 = np.array([0,0.034,0.067,0.1,0.134,0.167,0.2,0.234,0.267,0.3,0.334,0.367,
               0.4,0.434,0.467,0.5,0.534,0.567,0.6,0.634,0.667,0.7,0.733,
               0.767,0.8,0.833,0.867,0.9,0.933,0.967,1.0,1.033,1.067,1.1,
               1.133,1.167,1.2])
v3 = np.array([-0.55882353,-0.65671642,-0.66666667,-0.53731343,-0.64179104,
               -0.86363636,-0.91044776,-1.29850746,-1.59090909,-1.46268657,
               -1.59701493,-1.84848485,-1.8358209,-1.74626866,-1.77272727,
               -1.74626866,-1.79104478,-1.83333333,-1.7761194,-1.73134328,
               -1.83333333,-1.92424242,-1.88059701,-2.1641791,-2.18181818,
               -1.86567164,-1.89552239,-1.96969697,-1.94029851,-1.91044776,
               -2.0,-1.97014925,-2.01044776,-1.92424242,-1.97014925,
               -1.74328358,0.0874036])

# ------------------------------------------------------------------------
# 2. PROMEDIO ENTRE REPETICIONES Y DESVIACION ABSOLUTA MAXIMA (d.a.m.)
# ------------------------------------------------------------------------
t_avg = (t1 + t2 + t3) / 3.0
v_avg = (v1 + v2 + v3) / 3.0
V = np.vstack([v1, v2, v3])
dam = np.max(np.abs(V - v_avg), axis=0)


def ajuste_minimos_cuadrados(t, v):
    """Regresion lineal v = a*t + b por minimos cuadrados.
    Regresa a, b y sus incertidumbres (s_a, s_b) junto con R^2."""
    n = len(t)
    Sx, Sy = t.sum(), v.sum()
    Sxx, Sxy = (t**2).sum(), (t * v).sum()
    a = (n * Sxy - Sx * Sy) / (n * Sxx - Sx**2)
    b = (Sy - a * Sx) / n
    residuos = v - (a * t + b)
    s_y = np.sqrt(np.sum(residuos**2) / (n - 2))
    s_a = s_y * np.sqrt(n / (n * Sxx - Sx**2))
    s_b = s_y * np.sqrt(Sxx / (n * Sxx - Sx**2))
    r2 = 1 - np.sum(residuos**2) / np.sum((v - v.mean())**2)
    return a, b, s_a, s_b, r2


# ------------------------------------------------------------------------
# 3. DOS RANGOS DE AJUSTE A COMPARAR
#    (a) RANGO COMPLETO valido: t en [0, 1.099] s (excluye los ultimos 3
#        puntos, t >= 1.133 s, donde v cambia de signo por el impacto con
#        el piso / rebote).
#    (b) FASE DE ACELERACION INICIAL: t en [0, 0.433] s. A partir de ahi la
#        v_avg deja de crecer en magnitud de forma sostenida y oscila
#        alrededor de un valor casi constante (el objeto ya alcanzo su
#        velocidad terminal), por lo que un solo ajuste lineal sobre TODO
#        el rango subestima/mezcla ambos regimenes.
#    Ajustar N_COMPLETO / N_INICIAL si se revisa el criterio de corte.
# ------------------------------------------------------------------------
N_COMPLETO = 34
N_INICIAL = 14

a_c, b_c, sa_c, sb_c, r2_c = ajuste_minimos_cuadrados(t_avg[:N_COMPLETO], v_avg[:N_COMPLETO])
a_i, b_i, sa_i, sb_i, r2_i = ajuste_minimos_cuadrados(t_avg[:N_INICIAL], v_avg[:N_INICIAL])

print("=" * 70)
print("AJUSTE (a) - rango completo valido, t = 0 a %.3f s, N=%d" % (t_avg[N_COMPLETO-1], N_COMPLETO))
print(f"   a = {a_c:.3f} +/- {sa_c:.3f} m/s^2   b = {b_c:.3f} +/- {sb_c:.3f} m/s   R^2 = {r2_c:.3f}")
print(f"   a_exp (magnitud) = {abs(a_c):.1f} +/- {sa_c:.1f} m/s^2")
print("-" * 70)
print("AJUSTE (b) - solo fase de aceleracion inicial, t = 0 a %.3f s, N=%d" % (t_avg[N_INICIAL-1], N_INICIAL))
print(f"   a = {a_i:.3f} +/- {sa_i:.3f} m/s^2   b = {b_i:.3f} +/- {sb_i:.3f} m/s   R^2 = {r2_i:.3f}")
print(f"   a_exp (magnitud) = {abs(a_i):.1f} +/- {sa_i:.1f} m/s^2")
print("=" * 70)

# Velocidad terminal aproximada (tramo estable dentro del rango completo)
v_plateau = v_avg[N_INICIAL:N_COMPLETO]
v_term, v_term_std = v_plateau.mean(), v_plateau.std(ddof=1)
print(f"Velocidad terminal aproximada (t={t_avg[N_INICIAL]:.2f}-{t_avg[N_COMPLETO-1]:.2f} s): "
      f"v_t = {v_term:.2f} +/- {v_term_std:.2f} m/s")
print("=" * 70)

# ------------------------------------------------------------------------
# 4. GRAFICAS
# ------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# --- Figura 1: datos promedio con barras de error (d.a.m.) ---------------
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.errorbar(t_avg[:N_COMPLETO], v_avg[:N_COMPLETO], yerr=dam[:N_COMPLETO],
             fmt="o", ms=4, capsize=3, color="tab:blue", ecolor="tab:blue",
             elinewidth=1, label="Promedio de 3 repeticiones (valido)")
ax1.errorbar(t_avg[N_COMPLETO:], v_avg[N_COMPLETO:], yerr=dam[N_COMPLETO:],
             fmt="s", ms=5, capsize=3, color="tab:red", ecolor="tab:red",
             elinewidth=1, label="Impacto con el piso (excluido)")
ax1.axvline(t_avg[N_COMPLETO - 1], color="gray", ls=":", lw=1)
ax1.set_xlabel("t [s]")
ax1.set_ylabel("v [m/s]")
ax1.set_title("Filtro de cafe + 3 clips: velocidad promedio y barras de error (d.a.m.)")
ax1.legend()
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig("fig_con3clips_barras_error.png")

# --- Figura 2: comparacion de los dos ajustes por minimos cuadrados ------
fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
ax2.errorbar(t_avg[:N_COMPLETO], v_avg[:N_COMPLETO], yerr=dam[:N_COMPLETO],
             fmt="o", ms=4, capsize=3, color="tab:blue", ecolor="tab:blue",
             elinewidth=1, label="Promedio experimental (d.a.m.)", zorder=2)

t_line_c = np.linspace(0, t_avg[N_COMPLETO - 1], 200)
ax2.plot(t_line_c, a_c * t_line_c + b_c, color="black", lw=1.6, ls="--",
         label=f"(a) Rango completo: $v={a_c:.3f}t{b_c:+.3f}$  ($R^2$={r2_c:.2f})")

t_line_i = np.linspace(0, t_avg[N_INICIAL - 1] + 0.05, 100)
ax2.plot(t_line_i, a_i * t_line_i + b_i, color="tab:orange", lw=2.2,
         label=f"(b) Fase inicial: $v={a_i:.3f}t{b_i:+.3f}$  ($R^2$={r2_i:.2f})")

ax2.axhline(v_term, color="tab:green", ls=":", lw=1.3,
            label=f"$v_t$ aprox. $\\approx {v_term:.2f}$ m/s")

ax2.set_xlabel("t [s]")
ax2.set_ylabel("v [m/s]")
ax2.set_title("Filtro de cafe + 3 clips: ajuste por minimos cuadrados")
ax2.legend(fontsize=8.5, loc="lower left")
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("fig_con3clips_ajuste_mc.png")

print("Figuras guardadas: fig_con3clips_barras_error.png, "
      "fig_con3clips_ajuste_mc.png")
