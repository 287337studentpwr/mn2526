import numpy as np

# ------------------------------ pomocnicze ------------------------------


def print_result(name, result, exact=None):
    x, iterations = result

    print(name)
    print("wynik:", x)

    if exact is not None:
        print("dokladnie:", exact)
        print("blad:", abs(x - exact))

    print("iteracje:", iterations)
    print()


# ------------------------------ zad 1 ------------------------------

def bisection(a, b, F, n=10000, tolerance=1e-12, min_width=1e-10):
    if F(a) * F(b) > 0:
        raise ValueError("nie spelniono zalozenia")

    for i in range(n):
        c = (a + b) / 2

        if abs(b - a) < min_width:
            return c, i

        if abs(F(c)) <= tolerance:
            return c, i

        if F(c) * F(a) > 0:
            a = c
        else:
            b = c

    return (a + b) / 2, n

# ------------------------------ zad 2 ------------------------------

def newton(F, dF, x0, n=10000, tolerance=1e-12):
    x = x0

    for i in range(n):
        if abs(F(x)) <= tolerance:
            return x, i

        if dF(x) == 0:
            raise ValueError("dzielenie przez 0")

        x = x - F(x) / dF(x)

    return x, n

# ------------------------------ zad 3 ------------------------------

def exercise_3():
    print("------------------------------ zad 3 ------------------------------")

    # cos(x) = 0
    # dokladne rozwiazanie w przedziale (1, 2): pi/2

    F1 = lambda x: np.cos(x)
    dF1 = lambda x: -np.sin(x)
    exact1 = np.pi / 2

    print_result(
        "Bisekcja: cos(x) = 0",
        bisection(1, 2, F1),
        exact1
    )

    print_result(
        "Newton: cos(x) = 0",
        newton(F1, dF1, 1),
        exact1
    )

    # sin(x) = 0
    # dokladne rozwiazanie w przedziale (3, 4): pi

    F2 = lambda x: np.sin(x)
    dF2 = lambda x: np.cos(x)
    exact2 = np.pi

    print_result(
        "Bisekcja: sin(x) = 0",
        bisection(3, 4, F2),
        exact2
    )

    print_result(
        "Newton: sin(x) = 0",
        newton(F2, dF2, 3),
        exact2
    )

    # x^2 - 2 = 0
    # dokladne rozwiazanie w przedziale (1, 2): sqrt(2)

    F3 = lambda x: x**2 - 2
    dF3 = lambda x: 2*x
    exact3 = np.sqrt(2)

    print_result(
        "Bisekcja: x^2 - 2 = 0",
        bisection(1, 2, F3),
        exact3
    )

    print_result(
        "Newton: x^2 - 2 = 0",
        newton(F3, dF3, 1),
        exact3
    )


# ------------------------------ zad 4 ------------------------------

def exercise_4():
    print("------------------------------ zad 4 ------------------------------")

    # ln(x) = x / 6
    # F(x) = ln(x) - x/6

    F = lambda x: np.log(x) - x/6
    dF = lambda x: 1/x - 1/6

    print_result(
        "Bisekcja: pierwszy pierwiastek ln(x) = x/6",
        bisection(1, 2, F)
    )

    print_result(
        "Bisekcja: drugi pierwiastek ln(x) = x/6",
        bisection(10, 20, F)
    )

    print_result(
        "Newton: pierwszy pierwiastek ln(x) = x/6",
        newton(F, dF, 1.5)
    )

    print_result(
        "Newton: drugi pierwiastek ln(x) = x/6",
        newton(F, dF, 15)
    )


# ------------------------------ main ------------------------------

def main():
    exercise_3()
    exercise_4()


main()