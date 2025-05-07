import numpy as np
from numdifftools import Gradient
import matplotlib.pyplot as plt

# === Функция Химмельблау ===
def himmelblau(p):
    x, y = p
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# === Градиент функции Химмельблау ===
def grad_himmelblau(p):
    x, y = p
    df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([df_dx, df_dy])

# === Метод золотого сечения для выбора шага alpha ===
def opt_step(f, x, p, x1=0, x2=0.5, eps=1e-8):
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение ≈ 1.618
    a, b = x1, x2

    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi

    f1 = f(x + x1 * p)
    f2 = f(x + x2 * p)

    while abs(b - a) > eps:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) / phi
            f1 = f(x + x1 * p)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) / phi
            f2 = f(x + x2 * p)

    alpha = (a + b) / 2
    return alpha

# === Метод Бройдена ===
# === Метод Бройдена ===
def broyden_method(f, grad_f, x0, max_iter=1000, tol=1e-6, gamma_min=1e-8, gamma_max=1):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    d = len(x)
    B = np.eye(d)  # Начальная матрица — единичная
    g = grad_f(x)
    p = -B @ g  # Направление спуска

    iterations = 0  # Счётчик итераций

    for _ in range(max_iter):
        # Выбор шага методом золотого сечения
        gamma = opt_step(f, x, p)

        # Ограничение шага
        gamma = max(gamma_min, min(gamma, gamma_max))

        # Обновление точки
        x_new = x + gamma * p
        g_new = grad_f(x_new)

        # Проверка остановки по норме градиента
        if np.linalg.norm(g_new) < tol:
            print("Сошлось по градиенту")
            break

        # Обновление матрицы B по формуле Бройдена
        s = x_new - x
        y = g_new - g
        Bs = B @ s

        numerator = np.outer(y - Bs, s)
        denominator = np.dot(s, s)
        if denominator != 0:
            B += numerator / denominator

        # Регуляризация матрицы B
        B += np.eye(d) * 1e-6

        # Обновление направления спуска
        x = x_new
        g = g_new
        p = -B @ g

        # Сохранение истории
        path.append(x.copy())

        iterations += 1  # Увеличиваем счётчик итераций

    return x, path, iterations

# === Точка старта и запуск ===
x0 = [0, 0]  # Можно изменить начальную точку
result, path, iterations = broyden_method(himmelblau, grad_himmelblau, x0)

print("Найденный минимум:", result)
print("Значение функции в минимуме:", himmelblau(result))
print("Число итераций:", iterations)
# === Визуализация траектории ===
def plot_trajectory(path):
    path = np.array(path)
    x_coords = path[:, 0]
    y_coords = path[:, 1]

    # Диапазон значений для графика
    x_range = np.linspace(-6, 6, 400)
    y_range = np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = himmelblau([X, Y])

    known_minima = [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126]
    ]

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 30), cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Значение функции')
    plt.plot(x_coords, y_coords, 'o-', color='blue', label='Траектория метода')
    plt.plot(x_coords[0], y_coords[0], 'ro', markersize=8, label='Начальная точка')
    plt.plot(x_coords[-1], y_coords[-1], 'go', markersize=8, label='Найденный минимум')

    for xm, ym in known_minima:
        plt.plot(xm, ym, 'mx', markersize=10, label='Известный минимум')

    plt.title('Минимизация функции Химмельблау методом Бройдена')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# === Построение графика ===
plot_trajectory(path)