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
    B1 y B2 son los parámetros que cuantifican la aceptación del machismo para el hombre y la mujer.
'''

# t variable independiete


# Definimos la funcion
def sis_edos(t, ics,s1, s2, b1, b2, a1, a2):
    # Condiciones iniciales
    dv, di = ics[0], ics[1]

    edo1 = s1*((a1*b1) - dv)

    edo2 = s2*(((1-a2)*(1-b2)) - di)

    return [edo1, edo2]


# Parametros q
s1 = 0.25  # s1
s2 = 0.25  # s2
a1 = 0.5 # violencia observada en la infancia para el hombre
a2 = 0.6 # violencia observada en la infancia para la mujer
b1 = 0.3 # aceptación del machismo para el hombre
b2 = 0.3 # aceptación del machismo para la mujer

# intervalo donde se calcula la solucion
t0 = 0
tf = 10
t_span = np.array([t0, tf])

# Vector/arreglo con las condiciones iniciales
p0 = np.array([0.4, 0.2])

t = np.linspace(t0, tf, 100)

# resolviendo numericamente con solve_ivp
soln = solve_ivp(sis_edos, t_span, p0, t_eval=t, args=(s1, s2, b1, b2, a1, a2))
# print(soln)

# Extraer la solucion de la EDO1
x = soln.y[0, :]
# print(x)

# Extraer la solucion de la EDO2
y = soln.y[1, :]
# print(y)

# grafica
plt.plot(t, x, color="#86D2FF", linewidth=2.0, label="Índice de comportamiento violento del hombre")
plt.plot(t, y, color="#FF87D3", linewidth=2.0, label="Índice de independencia de la mujer")
plt.xlabel('Tiempo', fontsize=16, fontweight="bold")
plt.ylabel('Índice de agresión', fontsize=16, fontweight="bold")
plt.legend()
#plt.grid()
plt.title('Interacción de pareja íntima no influenciada ')
plt.show()