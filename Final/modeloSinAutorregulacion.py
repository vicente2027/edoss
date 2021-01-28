# -*- coding: utf-8 -*-
"""
@author: vicen
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

''' Sistema de ecuaciones a resolver 
    dv(t)/dt = S1(v? - v(t)) + k1(1-p1) (b1/y)i(t) + h(t), v(0) = v0
    di(t)/dt = S2(i? - i(t)) + k2(1-p2)uv(t), i(0) = i0      
    Donde 
    v(t) es el comportamiento violento del hombre en el tiempo t
    v(i) representa el estado(positivo) de libertad de la mujer
    o el potencial de dependencia(negativo)
    S1 y S2 son constantes positivas llamadas inercias
    k1 y k2 son constantes positivas 
    p1 y p2 son los parametros de auto regulacion para el hombre y la mujer respectivamente
    b1 y b2 son los parámetros que cuantifican la aceptación del machismo para el hombre y la mujer.
    y es la autoestima del hombre
    u es un factor externo como lo puede ser la familia o presion social
    h(t)son los eventos de consumo de alcohol 
'''


# t variable independiete

def sis_edos(t, ics,s1, s2, b1, b2, a1, a2, k1, k2, p1, p2, u, y, h):
    # Condiciones iniciales
    dv, di = ics[0], ics[1]

    #Modelo influenciado
    edo1 = s1*((a1*b1) - dv) + k1 * (1 - p1) * (b1 / y) * di + h
    edo2 = s2*(((1-a2)*(1-b2)) - di) + k2 * (1 - p2) * u * dv

    return [edo1, edo2]


# Parametros
s1 = 0.2  # s1
s2 = 0.25  # s2
a1 = 0.7 # violencia observada en la infancia para el hombre
a2 = 0.6 # violencia observada en la infancia para la mujer
b1 = 0.6# aceptación del machismo para el hombre
b2 = 0.5 # aceptación del machismo para la mujer
k1 = 1.0# constante positiva de proporcionalidad
k2 = 1.0 # constantes positivas de proporcionalidad
p1 = 0 # autorregulación para el hombre
p2 = 0.5 # autorregulación para la mujer
y = 0.2 # autoestima del hombre
u = -0.2 # factores externos como presión la social o la familia.
h = 0# eventos de consumo de alcohol
# intervalo donde se calcula la solucion
t0 = 0
tf = 30
t_span = np.array([t0, tf])

# Vector/arreglo con las condiciones iniciales
p0 = np.array([0.4, 0.2])

t = np.linspace(t0, tf, 100)

# resolviendo numericamente con solve_ivp
soln = solve_ivp(sis_edos, t_span, p0, t_eval=t, args=(s1, s2, b1, b2, a1, a2, k1, k2, p1, p2, u, y, h))
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
plt.title('Modelo sin  Consumo de Alcohol y sin autoregulacion del hombre')
#plt.grid()
plt.show()