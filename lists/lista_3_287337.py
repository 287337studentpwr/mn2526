import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# -------------------------------------------------- pomocnicze -------------------------------------------------
degrees_to_plot = [1, 2, 5, 10, 20] 

def f1(x):
    return np.abs(x)

def f2(x):
    return 1/(1 + 25*x**2)

def f5(x):
    return 40 + 10*x + 5*x**2 + 3*x**3 + 2*x**4 + x**5 + x**6
# -------------------------------------------------- zad 1 -------------------------------------------------


def solve_1_and_2(exercise, f_name, f):
    x = np.linspace(-1, 1, 50)
    y = f(x)

    x_test = np.linspace(-1, 1, 1000)
    y_test = f(x_test)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_test, label=f_name)

    for degree in range (1, 20 + 1):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        if degree in degrees_to_plot:
            plt.plot(x_test, poly(x_test), label=f"degree={degree}")

        error = np.max(np.abs(y_test - poly(x_test)))
        print(f"degree = {degree:2d}, max_error = {error:.10f}")
    
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title(f"exercise {exercise}")
    plt.legend()
    plt.show()

def exercise_1():
    exercise = 1
    f_name = "|x|"
    print("-------- exercise 1 --------\n")
    solve_1_and_2(exercise, f_name, f1)


# -------------------------------------------------- zad 2 -------------------------------------------------

def exercise_2():
    exercise = 2
    f_name = "1/(1+25x^2)"
    print("-------- exercise 2 --------\n")
    solve_1_and_2(exercise, f_name, f2)

# -------------------------------------------------- zad 3 -------------------------------------------------

def solve_3_and_4(exercise, f):
    test_points=2000
    
    x_test = np.linspace(-1, 1, test_points)
    y_test = f(x_test)

    errors = []


    for n in range(2, 20 + 1):
        x_nodes = np.linspace(-1, 1, n)
        y_nodes = f(x_nodes)

        spline = CubicSpline(x_nodes, y_nodes)
        y_approx = spline(x_test)
        

        max_error = np.max(np.abs(y_test - y_approx))
        
        errors.append((n, max_error))

        print(f"nodes = {n:2d}    max_error = {max_error:.10f}")

    return errors

def plot_errors(errors, exercise):
    nodes = [error[0] for error in errors]
    vals = [error[1] for error in errors]

    plt.figure(figsize=(8,5))
    plt.plot(nodes, vals, marker='o')
    plt.xlabel("number of nodes")
    plt.ylabel("max error")
    plt.title(f"exercise {exercise}")
    plt.grid(True)
    plt.show()

def exercise_3():
    exercise = 3
    print("-------- exercise 3 --------\n")
    errors = solve_3_and_4(exercise, f1)
    plot_errors(errors, exercise)
    
# -------------------------------------------------- zad 4 -------------------------------------------------
def exercise_4():
    exercise = 4
    print("-------- exercise 4 --------\n")
    errors = solve_3_and_4(exercise, f2)
    plot_errors(errors, exercise)
    
# -------------------------------------------------- zad 5 -------------------------------------------------

def solve_5():
    test_points=2000
    
    x_test = np.linspace(1, 7, test_points)
    y_test = f5(x_test)

    results = []

    for n in range(7, 15 + 1):
        x_nodes = np.linspace(1, 7, n)
        y_nodes = f5(x_nodes)

        coeffs = np.polyfit(x_nodes, y_nodes, 6)
        poly = np.poly1d(coeffs)

        y_approx = poly(x_test)
        max_error = np.max(np.abs(y_test - y_approx))

        results.append((n, max_error))
        print(f"nodes = {n:2d}    max_error = {max_error:.16e}")

    return results

def plot_task5(results):
    nodes = [result[0] for result in results]
    errs = [result[1] for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(nodes, errs, marker='o')
    plt.xlabel("number of nodes")
    plt.ylabel("Max error")
    plt.title("exercise 5")
    plt.grid(True)
    plt.show()

def exercise_5():
    print("-------- exercise 5 --------\n")
    
    results = solve_5()
    plot_task5(results)
    
    print("\nZwiekszenie liczby punktow kratowych nie poprawia zauwazalnie wyniku aproksymacji")
    
# -------------------------------------------------- main -------------------------------------------------
    
def main():
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

main()

    