import numpy as np
import matplotlib.pyplot as plt
# Функция Химмельблау
def himmelblau(p):
    x, y = p
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Градиент
def grad(p):
    x, y = p
    df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([df_dx, df_dy])

# Матрица Гессе
def hess(p):
    x, y = p
    H = np.zeros((2, 2))
    H[0, 0] = 4 * (x ** 2 + y - 11) + 8 * x ** 2 + 2
    H[0, 1] = H[1, 0] = 4 * x + 4 * y
    H[1, 1] = 2 + 4 * (x + y ** 2 - 7) + 8 * y ** 2
    return H

# Линейный поиск с условием Армихо
def line_search(f, grad, x, p, alpha=10, beta=0.5, sigma=1e-4, max_iter=10000):
    fx = f(x)
    g = grad(x)
    for _ in range(max_iter):
        if f(x + alpha * p) <= fx + sigma * alpha * np.dot(g, p):
            break
        alpha *= beta
    return alpha

# Метод Ньютона с линейным поиском
def newton_with_linesearch(f, grad, hess, x0, eps=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for k in range(max_iter):
        g = grad(x)
        H = hess(x)

        if np.linalg.norm(g) < eps:
            break

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Матрица Гессе вырожденная!")
            break

        d = -H_inv @ g  # Направление спуска

        alpha = line_search(f, grad, x, d)  # Подбор шага
        x = x + alpha * d  # Обновление точки с найденным шагом
        path.append(x.copy())
    return x, k+1, path

# Точка старта
x0 = [0, 0 ]

# Запуск
minimum, iterations, path = newton_with_linesearch(himmelblau, grad, hess, x0)

print(f"Найденная точка: {minimum}")
print(f"Значение функции: {himmelblau(minimum)}")
print(f"Количество итераций: {iterations}")


path = np.array(path)
x_coords = path[:, 0]
y_coords = path[:, 1]

x_range = np.linspace(-6, 6, 400)
y_range = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = himmelblau([X, Y])

minima = [
    [3.0, 2.0],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126]
]

plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 30), cmap='viridis', alpha=0.7)
plt.colorbar(contour, label='Значение функции')

plt.plot(x_coords, y_coords, 'o-', color='blue', label='Траектория метода')
plt.plot(x0[0], x0[1], 'ro', markersize=8, label='Начальная точка')
plt.plot(minimum[0], minimum[1], 'go', markersize=8, label='Найденный минимум')

for xm, ym in minima:
    plt.plot(xm, ym, 'mx', markersize=10, label='Известный минимум')

plt.title('Минимизация функции Химмельблау классическим методом Ньютона')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()