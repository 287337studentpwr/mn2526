import numpy as np

# ------------------------------ metody krokowe ------------------------------


def euler_step(y1, x1, x2, f):
    h = x2 - x1
    return y1 + h * f(x1, y1)


def heun_step(y1, x1, x2, f):
    # To jest metoda punktu środkowego, rząd 2
    h = x2 - x1
    return y1 + h * f(x1 + h/2, y1 + h/2 * f(x1, y1))


def rungekutty_step(y1, x1, x2, f):
    h = x2 - x1

    k1 = h * f(x1, y1)
    k2 = h * f(x1 + h/2, y1 + k1/2)
    k3 = h * f(x1 + h/2, y1 + k2/2)
    k4 = h * f(x1 + h, y1 + k3)

    return y1 + (k1 + 2*k2 + 2*k3 + k4) / 6


def solve_ode(step_method, f, x0, y0, x_end, n):
    xs = np.linspace(x0, x_end, n + 1)
    ys = np.zeros(n + 1)

    ys[0] = y0

    for i in range(n):
        ys[i + 1] = step_method(ys[i], xs[i], xs[i + 1], f)

    return xs, ys


# ------------------------------ zad 1 ------------------------------


def polynomial_test(step_method, method_name, p):
    x0 = 0
    y0 = 1
    x_end = 1
    n = 10

    print(method_name)
    print("-" * len(method_name))

    for degree in [p, p + 1]:

        def y_exact(x):
            return 1 + x**degree

        def f(x, y):
            return degree * x**(degree - 1)

        xs, ys = solve_ode(step_method, f, x0, y0, x_end, n)

        exact = y_exact(x_end)
        approx = ys[-1]
        error = abs(exact - approx)

        print(f"wielomian stopnia {degree}")
        print("approx:", approx)
        print("exact: ", exact)
        print("error: ", error)
        print()

    print()


def exercise_1():
    print("-------------- zad 1 --------------")
    methods = {
        "Euler": (euler_step, 1),
        "Heun / midpoint": (heun_step, 2),
        "Runge-Kutta 4": (rungekutty_step, 4),
    }

    for name, (method, p) in methods.items():
        polynomial_test(method, name, p)


# ------------------------------ zad 2 ------------------------------


def richardson_passive(xs, ys, f, step_method, p):

    n = len(xs)

    y1 = np.zeros(n)
    y2 = np.zeros(n)
    yr = np.zeros(n)

    y1[0] = ys[0]
    y2[0] = ys[0]
    yr[0] = ys[0]

    for i in range(n - 1):
        x_i = xs[i]
        x_next = xs[i + 1]
        h = x_next - x_i
        x_mid = x_i + h / 2

        # Jeden krok długości h
        y1[i + 1] = step_method(ys[i], x_i, x_next, f)

        # Dwa kroki długości h/2
        y_half = step_method(ys[i], x_i, x_mid, f)
        y2[i + 1] = step_method(y_half, x_mid, x_next, f)

        # Bierna poprawka Richardsona
        yr[i + 1] = (2**p * y2[i + 1] - y1[i + 1]) / (2**p - 1)

    return yr


def test_richardson_for_method(step_method, method_name, p):
    def f(x, y):
        return y

    def y_exact(x):
        return np.exp(x)

    x0 = 0
    y0 = 1
    x_end = 1
    n = 10

    xs, ys = solve_ode(step_method, f, x0, y0, x_end, n)
    yr = richardson_passive(xs, ys, f, step_method, p)

    exact = y_exact(x_end)

    basic_error = abs(exact - ys[-1])
    richardson_error = abs(exact - yr[-1])

    print(method_name)
    print("-" * len(method_name))
    print("metoda podstawowa:")
    print("approx:", ys[-1])
    print("error: ", basic_error)
    print()

    print("metoda + bierny Richardson:")
    print("approx:", yr[-1])
    print("error: ", richardson_error)
    print()

    print("poprawa błędu:")
    print(basic_error / richardson_error)
    print()
    print()


def exercise_2():
    print("-------------- zad 1 --------------")
    methods = {
        "Euler": (euler_step, 1),
        "Heun / midpoint": (heun_step, 2),
        "Runge-Kutta 4": (rungekutty_step, 4),
    }

    for name, (method, p) in methods.items():
        test_richardson_for_method(method, name, p)


# ------------------------------ zad 3 ------------------------------

def richardson_error_info(xs, ys, f, step_method, p, y_exact):
    n = len(xs)

    y1 = np.zeros(n)
    y2 = np.zeros(n)
    yr = np.zeros(n)

    estimated_error = np.zeros(n)
    real_error = np.zeros(n)

    y1[0] = ys[0]
    y2[0] = ys[0]
    yr[0] = ys[0]

    for i in range(n - 1):
        x_i = xs[i]
        x_next = xs[i + 1]
        h = x_next - x_i
        x_mid = x_i + h / 2

        # jeden krok h
        y1[i + 1] = step_method(ys[i], x_i, x_next, f)

        # dwa kroki h/2
        y_half = step_method(ys[i], x_i, x_mid, f)
        y2[i + 1] = step_method(y_half, x_mid, x_next, f)

        # wartość poprawiona Richardsonem
        yr[i + 1] = (2**p * y2[i + 1] - y1[i + 1]) / (2**p - 1)

        # informacja o błędzie z Richardsona
        estimated_error[i + 1] = abs(y2[i + 1] - y1[i + 1]) / (2**p - 1)

        # rzeczywisty błąd wartości poprawionej
        real_error[i + 1] = abs(y_exact(x_next) - yr[i + 1])

    return yr, estimated_error, real_error


def test_error_information_for_method(step_method, method_name, p):
    def f(x, y):
        return y

    def y_exact(x):
        return np.exp(x)

    x0 = 0
    y0 = 1
    x_end = 1
    n = 10

    xs, ys = solve_ode(step_method, f, x0, y0, x_end, n)

    yr, estimated_error, real_error = richardson_error_info(
        xs, ys, f, step_method, p, y_exact
    )

    print(method_name)
    print("-" * len(method_name))

    print("x        est_error        real_error       ratio")
    for i in range(1, len(xs)):
        ratio = estimated_error[i] / real_error[i] if real_error[i] != 0 else np.nan

        print(
            f"{xs[i]:.2f}    "
            f"{estimated_error[i]:.10e}    "
            f"{real_error[i]:.10e}    "
            f"{ratio:.5f}"
        )

    print()


def exercise_3():
    methods = {
        "Euler": (euler_step, 1),
        "Heun / midpoint": (heun_step, 2),
        "Runge-Kutta 4": (rungekutty_step, 4),
    }

    for name, (method, p) in methods.items():
        test_error_information_for_method(method, name, p)

# ------------------------------ zad 4 ------------------------------

def lorenz_system(t, u, sigma=10, r=28, b=2):
    x = u[0]
    y = u[1]
    z = u[2]

    dx = sigma * (y - x)
    dy = -x * z + r * x - y
    dz = x * y - b * z

    return np.array([dx, dy, dz])


def solve_system(step_method, f, t0, u0, t_end, n):
    ts = np.linspace(t0, t_end, n + 1)
    us = np.zeros((n + 1, len(u0)))

    us[0] = u0

    for i in range(n):
        us[i + 1] = step_method(us[i], ts[i], ts[i + 1], f)

    return ts, us


def exercise_4():
    t0 = 0
    t_end = 30
    n = 10000

    u0 = np.array([1.0, 1.0, 1.0])

    ts, us = solve_system(
        rungekutty_step,
        lorenz_system,
        t0,
        u0,
        t_end,
        n
    )

    print("Ostatnia wartość:")
    print("t =", ts[-1])
    print("x =", us[-1, 0])
    print("y =", us[-1, 1])
    print("z =", us[-1, 2]) 
# ------------------------------ uruchomienie ------------------------------

def main():
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

main()