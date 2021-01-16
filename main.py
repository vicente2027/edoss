# -*- coding: utf-8 -*-
"""
@author: vicen
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

''' Sistema de ecuaciones a resolver 
    dv(t)/dt = S1(v? - v(t)), v(0) = v0 # valor inicial 
    di(t)/dt = S2(i? - i(t)), i(0) = i0 # valor inicial     

    Donde 
    v(t) es el comportamiento violento del hombre en el tiempo t

    v(i) representa el estado(positivo) de libertad de la mujer
    o el potencial de dependencia(negativo)

    S1 y S2 son constantes positivas llamadas inercias
'''


# t variable independiete

def sis_edos(t, ics, v, i, s1, s2):
    # Condiciones iniciales
    dv, di = ics[0], ics[1]

    # Define una funcion para la 1era EDO s1' = phi1(t,s1,s2)
    edo1 = s1 * (v - dv)

    edo2 = s2 * (i - di)

    return [edo1, edo2]


# Parametros que defien la iteraccion de las dos especies
alfa = 30
beta = 1.0
gama = 1
delt = 1

# intervalo donde se calcula la solucion
t0 = 0
tf = 10
t_span = np.array([t0, tf])

# Vector/arreglo con las condiciones iniciales
p0 = np.array([0, 50])

t = np.linspace(t0, tf, 101)

# resolviendo numericamente con solve_ivp
soln = solve_ivp(sis_edos, t_span, p0, t_eval=t, args=(alfa, beta, gama, delt))
# print(soln)

# Extraer la solucion de la EDO1
x = soln.y[0, :]
# print(x)

# Extraer la solucion de la EDO2
y = soln.y[1, :]
# print(y)

# grafica
#plt.plot(t, x, ':r', linewidth=2.0, label="Man’s violent behavior index")
plt.plot(t, x, color="#86D2FF", linewidth=2.0, label="Índice de comportamiento violento del hombre")

plt.plot(t, y, color="#FF87D3", linewidth=2.0, label="Índice de independencia de la mujer")
plt.xlabel('Tiempo', fontsize=16, fontweight="bold")
plt.ylabel('Índice de agresión', fontsize=16, fontweight="bold")
plt.legend()
#plt.grid()
plt.title('Interacción de pareja íntima no influenciada ')
plt.show()