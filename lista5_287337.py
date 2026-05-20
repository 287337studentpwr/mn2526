import numpy as np

# ----------------------------- zad 1 ----------------------------- 

def solve_matrix_gauss_jordan(A, b):
    if A.ndim != 2:
        raise ValueError("Array must be 2d")
    n_rows, n_columns = A.shape
    if n_rows != n_columns:
        raise ValueError("2d Array must be a square matrix ")

    for i in range (n_columns):
        for j in range(n_rows):
            if A[i,i] == 0:
                raise ValueError("there can be no 0 on the diagonal")
            if j == i:
                continue
            if A[j, i] == 0:
                continue
            diff = A[j,i]/A[i,i]
            b[j] = b[j] - diff * b[i]  
            for ctn in range (i, n_columns): # mnozenie kazdego wiersza pewnie jest na to fkcja ale juz to napisalem, pomijam poprzednie kolumny bo powinny wyjsc zera
                A[j, ctn] = A[j, ctn] - diff * A[i, ctn]
    return b / np.diag(A)

def exercise_1():
    print("exercise 1\n")
    A1 = np.array([[2, 1],[1, 3]], dtype=float)
    b1 = np.array([5, 6], dtype=float)
    
    A2 = np.array([[2, -1, 1],[3, 3, 9],[3, 3, 5]], dtype=float)
    b2 = np.array([2, -1, 4], dtype=float)
    
    x1 = solve_matrix_gauss_jordan(A1, b1)
    print(A1 @ x1 - b1)  # jak wyjdzie 0 to git
    
    x1 = solve_matrix_gauss_jordan(A1, b1)
    print(A1 @ x1 - b1)  # jak wyjdzie 0 to git

# ----------------------------- zad 2 ----------------------------- 

def matrix_factorization(A):
    if A.ndim != 2:
        raise ValueError("Array must be 2d")
    n_rows, n_columns = A.shape
    if n_rows != n_columns:
        raise ValueError("2d Array must be a square matrix ")
    
    U = np.triu(A, k = 1)
    L = np.tril(A, k =-1)
    D = np.diag(np.diag(A))
    return L, D, U

def Jacobi_method(A, b, max_iter = 1000, tol = 1e-18):
    L, D, U = matrix_factorization(A)
    
    x_old = np.zeros_like(b, dtype = float)
    diagonal = np.diag(D)
    if 0 in diagonal:
        raise ValueError("no zero'es on diagonal")
    
    for i in range(max_iter):
        x_new = (-(L + U) @ x_old + b)/diagonal
        
        if  np.linalg.norm(x_new - x_old) < tol:
            return x_new
    
        x_old = x_new
        
    return x_old

def exercise_2():
    print("exercise 2\n")
    A1 = np.array([[2, 1],[1, 3]], dtype=float)
    b1 = np.array([5, 6], dtype=float)
    
    A2 = np.array([[2, -1, 1],[3, 3, 9],[3, 3, 5]], dtype=float)
    b2 = np.array([2, -1, 4], dtype=float)
    
    x1 = Jacobi_method(A1, b1)
    print(A1 @ x1 - b1)  # jak wyjdzie 0 to git
    
    x1 = Jacobi_method(A1, b1)
    print(A1 @ x1 - b1)  # jak wyjdzie 0 to git  
    
# ----------------------------- zad 3 ----------------------------- 

def exercise_3(epsilon1 = 1e-14, big_number = 100000000000000):
    print("exercise 3\n")
    jacobi_iter=6
    
    A1 = np.array([
        [4, 1, 1],
        [1, 4+epsilon1, 1],
        [1, 1, 4+epsilon1]
    ], dtype=float)
    b1 = np.array([6, 6+big_number, 6-big_number], dtype=float)
    
    A2 = np.array([
        [1, 1, 1],
        [1, 1+epsilon1, 1],
        [1, 1, 1+epsilon1]
    ], dtype=float)
    b2 = np.array([3, 3+big_number, 3-big_number], dtype=float)
    
    x_np1 = np.linalg.solve(A1.copy(), b1.copy())
    x_np2 = np.linalg.solve(A2.copy(), b2.copy())
    r_np1 = A1 @ x_np1 - b1
    r_np2 = A2 @ x_np2 - b2
    
    methods1 = [
        ("NumPy solve", np.linalg.solve(A1.copy(), b1.copy())),
        ("Gauss-Jordan", solve_matrix_gauss_jordan(A1.copy(), b1.copy())),
        ("Jacobi n", Jacobi_method(A1.copy(), b1.copy(), max_iter=1000)),
    ]
    
    methods2 = [
        ("NumPy solve", np.linalg.solve(A2.copy(), b2.copy())),
        ("Gauss-Jordan", solve_matrix_gauss_jordan(A2.copy(), b2.copy())),
        ("Jacobi n", Jacobi_method(A2.copy(), b2.copy(), max_iter=1000)),
    ]

    
    print("A1:\n")
    for name, x in methods1:
        r = A1 @ x - b1
        print(name)
        print("x =", x)
        print("||Ax - b|| =", np.linalg.norm(r))
        print("numpy diff=", np.linalg.norm(r - r_np1))
        print()
    
    print("A2:\n")
    for name, x in methods2:
        r = A2 @ x - b2
        print(name)
        print("x =", x)
        print("||Ax - b|| =", np.linalg.norm(r))
        print("numpy diff=", np.linalg.norm(r - r_np2))
        print()
    
    jacobi_comp = [
        ("Jacobi n", Jacobi_method(A1.copy(), b1.copy(), max_iter=jacobi_iter)),
        ("Jacobi n^2", Jacobi_method(A1.copy(), b1.copy(), max_iter=jacobi_iter**2)),
        ("Jacobi n^3", Jacobi_method(A1.copy(), b1.copy(), max_iter=jacobi_iter**3)),
    ]

    print(f"Jacobii, iter_ammount comparison, n={jacobi_iter}\n")
    for jacobi, x in jacobi_comp:
        r = A1 @ x - b1
        print(jacobi)
        print("x = ", x)
        print("||Ax - b|| =", np.linalg.norm(r))
        print()
        

# ----------------------------- main -----------------------------
 
def main():
    exercise_1()
    exercise_2()
    exercise_3()

main()       