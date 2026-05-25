import numpy as np

# -------------------------------------------------- zad 1 -------------------------------------------------
def exercise_1(): 
    A = np.array([
        [1, 1, 2, 1],
        [0, 1, 4, 3],
        [4, 6, 8, 6],
        [5, 5, -5, 5]
    ])

    B = np.array([
        [2, 1, 1, 2],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [2, 2, 2, 2]
    ])

    print(np.linalg.det(A))
    print(np.linalg.det(B))
    
    return None

# -------------------------------------------------- zad 2 -------------------------------------------------


def exercise_2():
    Small_Numbers = [10**(-i) for i in range(1, 15)]
    Big_Numbers = [10**(i) for i in range(1, 15)]
    r_norm_max = 0
    
    for epsilon1 in Small_Numbers:
        for epsilon2 in Big_Numbers:

            A = np.array([
                [1, 1, 1],
                [1, 1+epsilon1, 1],
                [1, 1, 1+epsilon1]
            ], dtype=float)

            b = np.array([3, 3+epsilon2, 3-epsilon2], dtype=float)

            x = np.linalg.solve(A, b)
            r = A @ x - b
            r_norm = np.linalg.norm(r)
            if r_norm_max < r_norm:
                r_max = r
                r_norm_max = r_norm
                x_max = x
                eps1 = epsilon1
                eps2 = epsilon2

    print("x =", x_max)
    print("r =", r_max)
    print("||r|| =", r_norm_max)
    print(f"epsilon1 = {eps1}, espilon2 = {eps2}")

    return None

# -------------------------------------------------- zad 2 -------------------------------------------------

def main():
    exercise_1()
    exercise_2()

main()
    