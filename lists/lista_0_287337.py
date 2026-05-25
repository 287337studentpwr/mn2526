import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
# -------------------------- zad 1 ----------------------------------
def calculate_y(n):
    L = [0 for _ in range(n+1)]
    L[0] = np.log(6/5)
    i = 1
    while (i<=n):
        L[i] = 1/i - 5*L[i-1]
        i += 1
    
    return L[n]

wyniki_sp_quad = [sp.quad((lambda x: (x**n)/(x+5)),0 ,1) for n in range(25)]

# -------------------------- zad 2 -------------------------------
def calculate_ex(x):
    z = 1
    sum = z
    for i in range(1, 11):
        z = z*(x/i)
        sum+= z
    return z

# -------------------------- plot -------------------------------
def plot(x):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(x, calculate_ex(x), label ="our f")
    plt.plot(x, np.exp(x), label = "np.exp")
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()
    plt.show()

# -------------------------- executables -------------------------------
wynik, _ = wyniki_sp_quad[24]
nasz_wynik = calculate_y(10)
blad = nasz_wynik - wynik

print(f"nasz wynik:  {nasz_wynik}")
print(f" wynik: {wynik}")
print(f"błąd: {blad} ")

x = np.linspace(-10, 0, 100)
plot(x)