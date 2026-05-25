import numpy as np

# ------------------------------ zad 1 ------------------------------

def trapeze_method(f, t0, y0, T, n=10):
    t = np.linspace(t0, T, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    
    for i in range(n):
        h = t[i + 1] - t[i]
        y[i + 1] = y[i] + h * (f(t[i]) + f(t[i + 1])) / 2

    return t, y

def exercise_1():
    print("zadanie_1")
    print("----------------------------------------")

    t0 = 0
    T = 1
    n = 10

    for degree in range (0,4):
        exact = lambda t, degree=degree: t**degree

        if degree == 0:
            f = lambda t: 0
            y0 = 1
        else:
            f = lambda t, degree=degree: degree * t**(degree - 1)
            y0 = 0

        t, y = trapeze_method(f, t0, y0, T, n)

        exact_values = exact(t)
        error = np.max(np.abs(y - exact_values))

        print(f"stopień {degree}, błąd = {error}")

# ------------------------------ zad 2 ------------------------------

def parabola_method(f, t0, y0, T, n):
    t = np.linspace(t0, T, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        h = t[i + 1] - t[i]

        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h, y[i] - h * k1 + 2 * h * k2)

        y[i + 1] = y[i] + h / 6 * (k1 + 4 * k2 + k3)

    return t, y

def exercise_2():
    print("\n")
    print("zadanie_2")
    print("--------------------------------")

    t0 = 0
    T = 1
    n = 10

    for degree in range (0,6):
        exact = lambda t, degree=degree: t**degree

        if degree == 0:
            f = lambda t, y: 0
            y0 = 1
        else:
            f = lambda t, y, degree=degree: degree * t**(degree - 1)
            y0 = 0

        t, y = parabola_method(f, t0, y0, T, n)
        error = np.max(np.abs(y - exact(t)))

        print(f"stopień {degree}, błąd = {error}")



# ------------------------------ zad 3 ------------------------------

def romberg_step(f, a, b, m):
    R = np.zeros((m + 1, m + 1))

    for k in range(m + 1):
        n = 2**k
        x = np.linspace(a, b, n + 1)
        h = (b - a) / n

        trapeze_sum = 0.5 * f(x[0]) + 0.5 * f(x[-1])
        for i in range(1, n):
            trapeze_sum += f(x[i])

        R[k, 0] = h * trapeze_sum

    for j in range(1, m + 1):
        for k in range(j, m + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)

    return R[m, m]

def romberg_method(f, t0, y0, T, n, m):
    t = np.linspace(t0, T, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + romberg_step(f, t[i], t[i + 1], m)

    return t, y

def exercise_3():
    print("\n")
    print("zadanie_3")
    print("--------------------------")

    t0 = 0
    T = 1
    n = 10
    m = 2   # poziom Romberga R[2,2]

    for degree in range(0, 9):
        exact = lambda t, degree=degree: t**degree

        if degree == 0:
            f = lambda t: 0
            y0 = 1
        else:
            f = lambda t, degree=degree: degree * t**(degree - 1)
            y0 = 0

        t, y = romberg_method(f, t0, y0, T, n, m)
        error = np.max(np.abs(y - exact(t)))

        print(f"stopień {degree}, błąd = {error}")

# ------------------------------ zad 4 ------------------------------

def rmoberg_step_2(f, a, b, m): # zmodyfikowany romberg step tak aby zwracał całe R
    R = np.zeros((m + 1, m + 1))

    for k in range(m + 1):
        n = 2**k
        x = np.linspace(a, b, n + 1)
        h = (b - a) / n

        trapeze_sum = 0.5 * f(x[0]) + 0.5 * f(x[-1])

        for i in range(1, n):
            trapeze_sum += f(x[i])

        R[k, 0] = h * trapeze_sum

    for j in range(1, m + 1):
        for k in range(j, m + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)

    return R


def romberg_integral_with_error(f, a, b, m):
    R = rmoberg_step_2(f, a, b, m)
    approx = R[m, m]
    error_estimate = abs(R[m, m] - R[m - 1, m - 1])

    return approx, error_estimate, R


def exercise_4():
    print("\n")
    print("zadanie_4")
    print("--------------------------------------------------------")

    tests = [
        {
            "name": "sin(x), [0, pi]",
            "f": lambda x: np.sin(x),
            "a": 0,
            "b": np.pi,
            "exact": 2
        },
        {
            "name": "exp(x), [0, 1]",
            "f": lambda x: np.exp(x),
            "a": 0,
            "b": 1,
            "exact": np.e - 1
        },
        {
            "name": "sqrt(x), [0, 1]",
            "f": lambda x: np.sqrt(x),
            "a": 0,
            "b": 1,
            "exact": 2 / 3
        }
    ]

    for m in range(1, 4):
        print(f"\nPoziom Romberga m = {m}")
        print("--------------------------------------------------------")

        for test in tests:
            approx, estimated_error, R = romberg_integral_with_error(
                test["f"],
                test["a"],
                test["b"],
                m
            )

            exact_error = abs(approx - test["exact"])

            if exact_error != 0:
                ratio = estimated_error / exact_error
            else:
                ratio = np.inf

            print(test["name"])
            print(f"  błąd rzeczywisty     = {exact_error:.3e}")
            print(f"  oszacowanie błędu    = {estimated_error:.3e}")
            print(f"  est_error / true_err = {ratio:.3e}")
            
# ------------------------------ zad 5 ------------------------------

def adaptive_romberg(f, a, b, tol=1e-8, max_m=10):

    previous_approx = None

    for m in range(max_m + 1):
        R = rmoberg_step_2(f, a, b, m)
        approx = R[m, m]

        if previous_approx is None:
            estimated_error = np.inf
        else:
            estimated_error = abs(approx - previous_approx)

        if estimated_error < tol:
            return approx, estimated_error, m, R

        previous_approx = approx

    raise RuntimeError("Nie osiągnięto zadanej dokładności.")


def exercise_5():
    print("\n")
    print("Zadanie 5: adaptacyjna metoda Romberga")
    print("--------------------------------------")

    tests = [
        {
            "name": "sin(x), [0, pi]",
            "f": lambda x: np.sin(x),
            "a": 0,
            "b": np.pi,
            "exact": 2
        },
        {
            "name": "exp(x), [0, 1]",
            "f": lambda x: np.exp(x),
            "a": 0,
            "b": 1,
            "exact": np.e - 1
        },
        {
            "name": "1 / (1 + x^2), [0, 1]",
            "f": lambda x: 1 / (1 + x**2),
            "a": 0,
            "b": 1,
            "exact": np.pi / 4
        },
        {
            "name": "sqrt(x), [0, 1]",
            "f": lambda x: np.sqrt(x),
            "a": 0,
            "b": 1,
            "exact": 2 / 3
        },
        {
            "name": "log(1 + x), [0, 1]",
            "f": lambda x: np.log(1 + x),
            "a": 0,
            "b": 1,
            "exact": 2 * np.log(2) - 1
        }
    ]

    tol = 1e-8

    for test in tests:
        approx, estimated_error, m, R = adaptive_romberg(
            test["f"],
            test["a"],
            test["b"],
            tol=tol,
            max_m=20
        )

        exact_error = abs(approx - test["exact"])

        print(test["name"])
        print(f"  wynik Romberga     = {approx:.15f}")
        print(f"  wynik dokładny     = {test['exact']:.15f}")
        print(f"  błąd rzeczywisty   = {exact_error:.3e}")
        print(f"  oszacowanie błędu  = {estimated_error:.3e}")
        print(f"  osiągnięty poziom  = m = {m}")
        print()
# ------------------------------ main ------------------------------
def main():
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    
main()
        
        
        