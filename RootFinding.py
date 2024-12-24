import numpy as np
import matplotlib.pyplot as plt

# Secant Method
def secant_method(f, x0, x1, tol=1e-7, max_iter=30):
    for iteration in range(1, max_iter + 1):
        fx0, fx1 = f(x0), f(x1)
        if fx1 - fx0 == 0:
            raise ValueError("Division by zero in Secant Method.")
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        rel_error = abs((x2 - x1) / x2)
        if rel_error < tol:
            return x2, f(x2), rel_error, iteration
        x0, x1 = x1, x2
    return x2, f(x2), rel_error, max_iter

# Bisection Method
def bisection_method(f, xl, xu, tol=0.1, max_iter=20):
    if f(xl) * f(xu) > 0:
        raise ValueError("No sign change detected in the interval.")
    for iteration in range(1, max_iter + 1):
        xr = (xl + xu) / 2
        if f(xl) * f(xr) < 0:
            xu = xr
        else:
            xl = xr
        rel_error = abs((xu - xl) / xr)
        if rel_error < tol:
            return xr, f(xr), rel_error, iteration
    return xr, f(xr), rel_error, max_iter

# False Position Method
def false_position_method(f, xl, xu, tol=0.1, max_iter=20):
    if f(xl) * f(xu) > 0:
        raise ValueError("No sign change detected in the interval.")
    for iteration in range(1, max_iter + 1):
        xr = xu - (f(xu) * (xl - xu)) / (f(xl) - f(xu))
        if f(xl) * f(xr) < 0:
            xu = xr
        else:
            xl = xr
        rel_error = abs((xu - xl) / xr)
        if rel_error < tol:
            return xr, f(xr), rel_error, iteration
    return xr, f(xr), rel_error, max_iter

# Newton-Raphson Method
def newton_method(f, df, x0, tol=1e-7, max_iter=30):
    for iteration in range(1, max_iter + 1):
        fx, dfx = f(x0), df(x0)
        if dfx == 0:
            raise ValueError("Derivative is zero; Newton's method fails.")
        x1 = x0 - fx / dfx
        rel_error = abs((x1 - x0) / x1)
        if rel_error < tol:
            return x1, f(x1), rel_error, iteration
        x0 = x1
    return x1, f(x1), rel_error, max_iter

# Functions for Part 1
functions_part1 = {
    "-x^2 + 2x + 2": lambda x: -x**2 + 2*x + 2,
    "e^x + x - 7": lambda x: np.exp(x) + x - 7,
    "e^x + sin(x) - 4": lambda x: np.exp(x) + np.sin(x) - 4
}

# Functions for Part 2
functions_part2 = {
    "x^3 - 9": lambda x: x**3 - 9,
    "3x^3 + x^2 - x - 5": lambda x: 3 * x**3 + x**2 - x - 5,
    "cos(x)^2 - x + 6": lambda x: np.cos(x)**2 - x + 6
}

# Function for Part 3
f_part3 = lambda x: 7 * np.sin(x) * np.exp(-x) - 1
df_part3 = lambda x: 7 * (np.cos(x) * np.exp(-x) - np.sin(x) * np.exp(-x))

# Part 1: Secant Method
print("\ The Secant Method Solution")
results_part1 = []
for name, func in functions_part1.items():
    try:
        root, func_value, rel_error, iterations = secant_method(func, 1, 2)
        results_part1.append((name, root, func_value, rel_error, iterations))
        print(f"Function: {name}")
        print(f"  Solution = {root:.7f}")
        print(f"  Function value at solution = {func_value:.7e}")
        print(f"  Relative error = {rel_error:.7e}")
        print(f"  Number of iterations = {iterations}")
    except ValueError as e:
        print(f"  Error for {name}: {e}")

# Part 2: Bisection and False Position Methods
print("\nThe Bisection and False Position Solution")
results_part2 = []
for name, func in functions_part2.items():
    try:
        print(f"Function: {name}")
        # Bisection
        root_b, func_b, rel_b, iter_b = bisection_method(func, 1, 10)
        # False Position
        root_fp, func_fp, rel_fp, iter_fp = false_position_method(func, 1, 10)
        results_part2.append((name, root_b, func_b, rel_b, iter_b, root_fp, func_fp, rel_fp, iter_fp))
        print(f"  Bisection Method:")
        print(f"    Solution = {root_b:.7f}")
        print(f"    Function value at solution = {func_b:.7e}")
        print(f"    Relative error = {rel_b:.7e}")
        print(f"    Number of iterations = {iter_b}")
        print(f"  False Position Method:")
        print(f"    Solution = {root_fp:.7f}")
        print(f"    Function value at solution = {func_fp:.7e}")
        print(f"    Relative error = {rel_fp:.7e}")
        print(f"    Number of iterations = {iter_fp}")
    except ValueError as e:
        print(f"  Error for {name}: {e}")

# Part 3: Newton-Raphson Method
print("\nThe Newton-Raphson Method Solution")
x0_part3 = 0.3
try:
    root, func_value, rel_error, iterations = newton_method(f_part3, df_part3, x0_part3)
    print(f"  Solution = {root:.7f}")
    print(f"  Function value at solution = {func_value:.7e}")
    print(f"  Relative error = {rel_error:.7e}")
    print(f"  Number of iterations = {iterations}")
except ValueError as e:
    print(f"  Error: {e}")

# Graphing Part 1 and Part 2
x_vals = np.linspace(-10, 10, 1000)
plt.figure(figsize=(12, 6))

# Plot Part 1 Functions
for name, func in functions_part1.items():
    plt.plot(x_vals, func(x_vals), label=f"Part 1: {name}")

# Plot Part 2 Functions
for name, func in functions_part2.items():
    plt.plot(x_vals, func(x_vals), label=f"Part 2: {name}")

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Parts 1 and 2: Root-Finding Method Visualization - Jackson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# Graphing Part 3
plt.figure(figsize=(12, 6))
plt.plot(x_vals, f_part3(x_vals), label="Part 3: 7sin(x)e^-x - 1", color="blue")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

plt.title("Part 3:Newton Method Visualization - Jackson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# Show All Graphs
plt.show()
