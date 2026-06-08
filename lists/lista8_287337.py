import numpy as np
import matplotlib.pyplot as plt

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


def simplex_average_size(points):
    distances = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(np.linalg.norm(points[i] - points[j]))

    return sum(distances) / len(distances)


def sort_points(f, points):
    points = sorted(points, key=f)
    return np.array(points)


def nelder_mead(f, x0, epsilon=1e-6, max_iter=10000):
    x0 = np.array(x0, dtype=float)

    x1 = np.array([x0[0] + 1, x0[1]])
    x2 = np.array([x0[0], x0[1] + 1])

    points = np.array([x0, x1, x2])
    iterations = 0

    while simplex_average_size(points) > epsilon and iterations < max_iter:
        iterations += 1

        points = sort_points(f, points)

        best = points[0]
        mid = points[1]
        worst = points[2]

        middle_point = (best + mid) / 2

        # odbicie
        reflection = 2 * middle_point - worst

        if f(reflection) < f(mid):
            if f(reflection) < f(best):
                # ekspansja
                better_reflection = middle_point + 2 * (reflection - middle_point)

                if f(better_reflection) < f(reflection):
                    worst = better_reflection
                else:
                    worst = reflection
            else:
                worst = reflection

        else:
            # kontrakcja
            p1 = (worst + middle_point) / 2
            p2 = (reflection + middle_point) / 2

            if f(p2) < f(p1):
                p1 = p2

            if f(p1) < f(worst):
                worst = p1

            else:
                # zmniejszenie trójkąta
                mid = (mid + best) / 2
                worst = (worst + best) / 2

        points = np.array([best, mid, worst])

    points = sort_points(f, points)
    best = points[0]

    return best, f(best), iterations


# ------------------------------ metoda najszybszego spadku ------------------------------


def steepest_descent(f, grad_f, x0, epsilon=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    iterations = 0

    while np.linalg.norm(grad_f(x)) > epsilon and iterations < max_iter:
        grad = grad_f(x)

        # kierunek największego spadku
        direction = -grad

        alpha = 1.0

        # proste szukanie kroku
        while f(x + alpha * direction) >= f(x):
            alpha /= 2

            if alpha < 1e-12:
                break

        x = x + alpha * direction
        iterations += 1

    return x, f(x), iterations


# ------------------------------ metoda BFGS ------------------------------


def bfgs(f, grad_f, x0, epsilon=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    n = len(x)

    # V to przybliżenie odwrotności hesjanu
    V = np.eye(n)

    iterations = 0

    while np.linalg.norm(grad_f(x)) > epsilon and iterations < max_iter:
        grad = grad_f(x)

        # kierunek BFGS
        direction = -V @ grad

        # zabezpieczenie: jeśli kierunek nie jest kierunkiem spadku, wracamy do zwykłego największego spadku
        if grad @ direction >= 0:
            V = np.eye(n)
            direction = -grad

        alpha = 1.0

        # proste szukanie kroku
        while f(x + alpha * direction) >= f(x):
            alpha /= 2

            if alpha < 1e-12:
                break

        x_new = x + alpha * direction
        grad_new = grad_f(x_new)

        r = x_new - x # zmiana punktu
        s = grad_new - grad # zmiana gradientu

        denominator = s @ r

        # aktualizacja BFGS
        if denominator > 1e-12:
            Vs = V @ s

            V = V + (
                (1 + (s @ Vs) / denominator) * np.outer(r, r) / denominator
                - (np.outer(Vs, r) + np.outer(r, Vs)) / denominator
            )

        x = x_new
        iterations += 1

    return x, f(x), iterations


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
        print()
        print("Funkcja", function_name)
        print("-" * 40)

        for method_name, method in no_gradient_methods.items():
            test_method_on_function(method, method_name, function_name, f)

        for method_name, method in gradient_methods.items():
            test_method_on_function(method, method_name, function_name, f, grad_f)


# ------------------------------ zad 2 - zewnętrzna funkcja kary ------------------------------


def g_circle(x):
    # g(x) = x[0]^2 + x[1]^2 - 1
    
    return x[0]**2 + x[1]**2 - 1


def external_penalty_function(f, constraints, rho):
    # Kara pojawia się tylko wtedy, gdy punkt wychodzi poza obszar dopuszczalny.
    
    def f_penalty(x):
        penalty = 0.0

        for g in constraints:
            penalty += max(0, g(x))**2 # jezeli jest w kole to jest ujemne wiec wezmiemy 0

        return f(x) + rho * penalty

    return f_penalty


def external_penalty_method(
    f,
    constraints,
    x0,
    method,
    rho_start=1.0,
    rho_multiplier=10.0,
    epsilon=1e-6,
    max_outer_iter=20
):

    x = np.array(x0, dtype=float)
    rho = rho_start

    iterations = 0

    for i in range(max_outer_iter):
        f_penalty = external_penalty_function(f, constraints, rho)

        x_new, f_penalty_min, inner_iterations = method(f_penalty, x)

        iterations += inner_iterations

        if np.linalg.norm(x_new - x) < epsilon:
            x = x_new
            break

        x = x_new
        rho *= rho_multiplier

    return x, f(x), iterations, rho

def plot_exercise_2(x_min):
    # koło x^2 + y^2 <= 1
    t = np.linspace(0, 2 * np.pi, 400)
    circle_x = np.cos(t)
    circle_y = np.sin(t)

    # zwykłe minimum funkcji f_1b bez ograniczeń
    unconstrained_min = np.array([1.0, -2.0])

    plt.figure(figsize=(7, 7))

    # obszar dopuszczalny
    plt.plot(circle_x, circle_y, label="brzeg obszaru: x^2 + y^2 = 1")
    plt.fill(circle_x, circle_y, alpha=0.15)

    # zwykłe minimum
    plt.scatter(
        unconstrained_min[0],
        unconstrained_min[1],
        marker="x",
        s=120,
        label="minimum bez ograniczeń"
    )

    # nasze minimum z karą
    plt.scatter(
        x_min[0],
        x_min[1],
        marker="o",
        s=80,
        label="minimum z ograniczeniem"
    )

    # opisy punktów
    plt.text(
        unconstrained_min[0] + 0.05,
        unconstrained_min[1],
        "(1, -2)"
    )

    plt.text(
        x_min[0] + 0.05,
        x_min[1],
        f"({x_min[0]:.3f}, {x_min[1]:.3f})"
    )

    # osie
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlim(-1.5, 1.5)
    plt.ylim(-2.3, 1.3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Zadanie 2: minimum funkcji f_1b w kole jednostkowym")
    plt.legend()
    plt.grid(True)

    plt.show()



def exercise_2():
    print("-------------- zad 2 --------------")

    print("Wybrana funkcja: f_1b")
    print("Ograniczenie: x^2 + y^2 <= 1")
    print("Metoda: zewnętrzna funkcja kary + Nelder-Mead")
    print()

    x0 = np.array([0.5, 0.5])

    constraints = [g_circle]

    x_min, f_min, iterations, rho = external_penalty_method(
        f=f_1b,
        constraints=constraints,
        x0=x0,
        method=nelder_mead,
        rho_start=1.0,
        rho_multiplier=10.0,
        epsilon=1e-6,
        max_outer_iter=20
    )

    print("x_min:", x_min)
    print("f_min:", f_min)
    print("g(x_min):", g_circle(x_min))
    print("iterations:", iterations)
    print("rho:", rho)

    if g_circle(x_min) <= 1e-6:
        print("Punkt spełnia ograniczenie.")
    else:
        print("Punkt NIE spełnia ograniczenia.")

    print()
    print("Dla porównania zwykłe minimum funkcji f_1b bez ograniczeń to punkt [1, -2].")
    print("Ten punkt nie należy do koła x^2 + y^2 <= 1, więc minimum z ograniczeniem leży na brzegu obszaru.")
    plot_exercise_2(x_min)

exercise_1()
exercise_2()