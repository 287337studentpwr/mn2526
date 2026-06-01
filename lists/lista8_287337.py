import numpy as np

# ------------------------------ pomocnicze ------------------------------


def norm(x):
    return np.linalg.norm(x)


def numerical_gradient(f, x, h=1e-6):
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2 * h)

    return grad


def golden_section_search(phi, a=0.0, b=1.0, tolerance=1e-7, max_iter=200):
    gr = (np.sqrt(5) - 1) / 2

    c = b - gr * (b - a)
    d = a + gr * (b - a)

    fc = phi(c)
    fd = phi(d)

    for _ in range(max_iter):
        if abs(b - a) < tolerance:
            break

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = phi(d)

    return (a + b) / 2


# ------------------------------ funkcje z zadania ------------------------------


def f_1a(x):
    return np.sin(x[0]) + np.cos(x[0])


def grad_1a(x):
    return np.array([np.cos(x[0]) - np.sin(x[0]), 0.0])


def f_1b(x):
    return (x[0] - 1)**2 + (x[1] + 2)**2 + 3


def grad_1b(x):
    return np.array([
        2 * (x[0] - 1),
        2 * (x[1] + 2)
    ])


def f_1c(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


def grad_1c(x):
    return np.array([
        4*x[0] - 4.2*x[0]**3 + x[0]**5 + x[1],
        x[0] + 2*x[1]
    ])


def f_1d(x):
    return np.exp(x[0]) - x[0] + np.exp(x[1]) - x[1]


def grad_1d(x):
    return np.array([
        np.exp(x[0]) - 1,
        np.exp(x[1]) - 1
    ])


# ------------------------------ metoda Hooke'a-Jeevesa ------------------------------


def hooke_jeeves(f, x0, delta=1.0, epsilon=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    n = len(x)
    iterations = 0

    while delta >= epsilon and iterations < max_iter:
        improved = False

        for i in range(n):
            direction = np.zeros(n)
            direction[i] = 1.0

            if f(x + delta * direction) < f(x):
                x = x + delta * direction
                improved = True
            elif f(x - delta * direction) < f(x):
                x = x - delta * direction
                improved = True

        if not improved:
            delta /= 2

        iterations += 1

    return x, f(x), iterations


# ------------------------------ metoda Neldera-Meada ------------------------------


def nelder_mead(f, x0, step=1.0, epsilon=1e-6, max_iter=10000):
    x0 = np.array(x0, dtype=float)
    n = len(x0)

    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] += step
        simplex.append(x)

    simplex = np.array(simplex)

    for iteration in range(max_iter):
        values = np.array([f(x) for x in simplex])
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        best = simplex[0]
        worst = simplex[-1]
        second_worst_value = values[-2]

        diameter = max(norm(simplex[i] - best) for i in range(n + 1))
        if diameter < epsilon:
            return best, f(best), iteration

        c = np.mean(simplex[:-1], axis=0)

        xr = c + (c - worst)
        fr = f(xr)

        if fr < values[1]:
            xs = c + 2 * (c - worst)
            if f(xs) < fr:
                simplex[-1] = xs
            else:
                simplex[-1] = xr

        elif values[1] <= fr < second_worst_value:
            simplex[-1] = xr

        elif second_worst_value <= fr < values[-1]:
            xz = c + 0.5 * (c - worst)
            if f(xz) < fr:
                simplex[-1] = xz
            else:
                for i in range(1, n + 1):
                    simplex[i] = best + 0.5 * (simplex[i] - best)

        else:
            xw = c - 0.5 * (c - worst)
            if f(xw) < values[-1]:
                simplex[-1] = xw
            else:
                for i in range(1, n + 1):
                    simplex[i] = best + 0.5 * (simplex[i] - best)

    values = np.array([f(x) for x in simplex])
    best = simplex[np.argmin(values)]
    return best, f(best), max_iter


# ------------------------------ metoda najszybszego spadku ------------------------------


def steepest_descent(f, grad_f, x0, epsilon=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)

    for iteration in range(max_iter):
        grad = grad_f(x)

        if norm(grad) < epsilon:
            return x, f(x), iteration

        direction = -grad

        def phi(t):
            return f(x + t * direction)

        t = golden_section_search(phi, 0, 1)
        x = x + t * direction

    return x, f(x), max_iter


# ------------------------------ metoda BFGS ------------------------------


def bfgs(f, grad_f, x0, epsilon=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    n = len(x)
    V = np.eye(n)

    for iteration in range(max_iter):
        grad = grad_f(x)

        if norm(grad) < epsilon:
            return x, f(x), iteration

        direction = -V @ grad

        def phi(t):
            return f(x + t * direction)

        t = golden_section_search(phi, 0, 1)
        x_new = x + t * direction
        grad_new = grad_f(x_new)

        r = x_new - x
        s = grad_new - grad
        rs = r @ s

        if abs(rs) > 1e-12:
            I = np.eye(n)
            V = (I - np.outer(r, s) / rs) @ V @ (I - np.outer(s, r) / rs) + np.outer(r, r) / rs
        else:
            V = np.eye(n)

        x = x_new

    return x, f(x), max_iter


# ------------------------------ zad 1 ------------------------------


def test_method_on_function(method, method_name, function_name, f, grad_f=None):
    x0 = np.array([0.5, 0.5])

    if grad_f is None:
        x_min, f_min, iterations = method(f, x0)
    else:
        x_min, f_min, iterations = method(f, grad_f, x0)

    print(method_name, "-", function_name)
    print("x_min:", x_min)
    print("f_min:", f_min)
    print("iterations:", iterations)
    print()


def exercise_1():
    print("-------------- zad 1 --------------")

    functions = {
        "a": (f_1a, grad_1a),
        "b": (f_1b, grad_1b),
        "c": (f_1c, grad_1c),
        "d": (f_1d, grad_1d),
    }

    no_gradient_methods = {
        "Hooke-Jeeves": hooke_jeeves,
        "Nelder-Mead": nelder_mead,
    }

    gradient_methods = {
        "Najszybszy spadek": steepest_descent,
        "BFGS": bfgs,
    }

    for function_name, (f, grad_f) in functions.items():
        for method_name, method in no_gradient_methods.items():
            test_method_on_function(method, method_name, function_name, f)

        for method_name, method in gradient_methods.items():
            test_method_on_function(method, method_name, function_name, f, grad_f)


# ------------------------------ zad 2 ------------------------------

# Numer albumu 287337 jest nieparzysty, więc wybieram metodę wewnętrznej funkcji kary.
# Minimalizuję funkcję z punktu 1b na kole x^2 + y^2 <= 1.
# Minimum bez ograniczeń jest w punkcie (1, -2), czyli poza kołem,
# więc minimum z ograniczeniem powinno leżeć na brzegu obszaru.


def circle_constraint(x):
    return x[0]**2 + x[1]**2 - 1


def internal_penalty_function(f, rho):
    def f_penalty(x):
        g = circle_constraint(x)

        if g >= 0:
            return 1e100

        return f(x) + rho / (-g)

    return f_penalty


def safe_bfgs_inside_circle(f, x0, epsilon=1e-6, max_iter=10000):
    def grad_f(x):
        return numerical_gradient(f, x)

    x = np.array(x0, dtype=float)
    n = len(x)
    V = np.eye(n)

    for iteration in range(max_iter):
        grad = grad_f(x)

        if norm(grad) < epsilon:
            return x, f(x), iteration

        direction = -V @ grad
        t_max = 1.0

        while circle_constraint(x + t_max * direction) >= 0:
            t_max /= 2
            if t_max < 1e-12:
                return x, f(x), iteration

        def phi(t):
            return f(x + t * direction)

        t = golden_section_search(phi, 0, t_max)
        x_new = x + t * direction
        grad_new = grad_f(x_new)

        r = x_new - x
        s = grad_new - grad
        rs = r @ s

        if abs(rs) > 1e-12:
            I = np.eye(n)
            V = (I - np.outer(r, s) / rs) @ V @ (I - np.outer(s, r) / rs) + np.outer(r, r) / rs
        else:
            V = np.eye(n)

        x = x_new

    return x, f(x), max_iter


def internal_penalty_method(f, x0, rhos, epsilon=1e-6):
    x = np.array(x0, dtype=float)
    previous_x = None

    for i, rho in enumerate(rhos):
        f_penalty = internal_penalty_function(f, rho)
        x, _, iterations = safe_bfgs_inside_circle(f_penalty, x)

        print("iteracja kary:", i + 1)
        print("rho:", rho)
        print("x:", x)
        print("f(x):", f(x))
        print("g(x):", circle_constraint(x))
        print("iterations:", iterations)
        print()

        if previous_x is not None and norm(x - previous_x) < epsilon:
            break

        previous_x = x.copy()

    return x, f(x)


def exercise_2():
    print("-------------- zad 2 --------------")

    x0 = np.array([0.0, 0.0])
    rhos = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]

    x_min, f_min = internal_penalty_method(f_1b, x0, rhos)

    print("wynik końcowy")
    print("x_min:", x_min)
    print("f_min:", f_min)
    print("g(x_min):", circle_constraint(x_min))
    print()


# ------------------------------ uruchomienie ------------------------------


def main():
    exercise_1()
    exercise_2()


main()
