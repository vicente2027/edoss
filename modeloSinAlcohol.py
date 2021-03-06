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

    v = (s1s2v?+s2k1(1-p1)(b1/y)i?)/(s1s2-k1k2(1 -p2)(b1/y)u), v(0) = v0 # valor inicial 

    i = (s1s2i?+s1k2(1-p2)uv?)/(s1s2-k1k2(1-p1)(1-p2)(b1/y)u), i0 # valor inicial 



    Donde 
    v(t) es el comportamiento violento del hombre en el tiempo t

    v(i) representa el estado(positivo) de libertad de la mujer
    o el potencial de dependencia(negativo)

    S1 y S2 son constantes positivas llamadas inercias

    k1 y k2 son constantes positivas 
    p1 y p2 son los parametros de auto regulacion para el hombre y la mujer respectivamente
    y es la autoestima del hombre
    u es un factor externo como lo puede ser la familia o presion social
'''


# t variable independiete

def sis_edos(t, ics, v, i, s1, s2, k1, k2, p1, p2, y, u, b1):
    # Condiciones iniciales
    dv, di = ics[0], ics[1]
    #edo1 = ((s1 * s2 * v) + (s2 * (k1 * (1 - p1) * (b1 / y)) * i)) / ((s1*s2)-k1*(k2*(1.0-p1)(1.0-p2)*(b1/y)*-u))
    edo1 = ((s1 * s2 * (a1*b1)) + (s2 * (k1 * (1 - p1) * (b1 / y)) * ((1-a2)*(1-b2)))) / (s1 * s2) - (k1 * (k2 * (1 - p1) * (1 - p2) * (b1 / y) * u))
    #edo2 = (s1*s2*-i+s1*k2*(1-p2)*u*-v) / (s1*s2-k1*k2*(1-p1)*(1-p2)*(b1/y)*u)
    edo2 = ((s1 * s2) * (((1-a2)*(1-b2)) + s1) * (k2 * (1 - p2)) * u * (a1*b1)) / (s1 * s2) - (k1 * k2 * (1 - p1) * (1 - p2) * (b1 / y) * u)
    return [edo1, edo2]


# Parametros que defien la iteraccion de las dos especies
vt = 0 # v(t) violence index
it = 0 # i(t) independece index
s1 = 0.25  # s1
s2 = 0.25  # s2
#k1 = 0.60
#k2 = 0.25
k1 = 1
k2 = 1
p1 = 0.50
p2 = 0.50
y = 0.60
u = 0.60
a1 = 0.5
a2 = 0.6
b1 = 0.3
b2 = 0.3
# intervalo donde se calcula la solucion
t0 = 0
tf = 20
t_span = np.array([t0, tf])

# Vector/arreglo con las condiciones iniciales
p0 = np.array([0.4, 0.2])

t = np.linspace(t0, tf, 100)

# resolviendo numericamente con solve_ivp
soln = solve_ivp(sis_edos, t_span, p0, t_eval=t, args=(vt, it, s1, s2, k1, k2, p1, p2, y, u, b1))
# print(soln)

# Extraer la solucion de la EDO1
x = soln.y[0, :]
# print(x)

# Extraer la solucion de la EDO2
y = soln.y[1, :]
# print(y)

# grafica

plt.plot(t, x, color="#86D2FF" , linewidth=2.0, label="Índice de comportamiento violento del hombre")
plt.plot(t, y, color="#FF87D3", linewidth=2.0, label="Índice de independencia de la mujer")
plt.xlabel('Tiempo', fontsize=16, fontweight="bold")
plt.ylabel('Índice de agresión ', fontsize=16, fontweight="bold")
#plt.ylim(-20, 80)
#plt.xlim(4, 5.8)


#plt.figure(figsize=(6, 4))
plt.legend()
plt.title('Modelo sin comsumo de Alcohol')
#lt.grid()
plt.show()