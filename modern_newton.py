import numpy as np
import matplotlib.pyplot as plt

def himmelblau(p):
    x, y = p
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def grad_himmelblau(p):
    x, y = p
    df_dx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
    df_dy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
    return np.array([df_dx, df_dy])

def hess_himmelblau(p):
    x, y = p
    H = np.zeros((2, 2))
    H[0, 0] = 4 * (x ** 2 + y - 11) + 8 * x ** 2 + 2
    H[0, 1] = H[1, 0] = 4 * x + 4 * y
    H[1, 1] = 2 + 4 * (x + y ** 2 - 7) + 8 * y ** 2
    return H

def armijo(f, grad, x, p, alpha=10, beta=0.5, sigma=1e-4, max_iter=100):
    fx = f(x)
    g = grad(x)
    for _ in range(max_iter):
        if f(x + alpha * p) <= fx + sigma * alpha * np.dot(g, p):
            break
        alpha *= beta
    return alpha

def newton_method(f, grad, hess, x0, max_iter=500):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    
    # Вычисляем Гессиан только один раз
    H_fixed = hess(x)
    
    # Регуляризуем матрицу Гессе (добавляем малое значение на диагональ)
    epsilon = 1e-6
    H_fixed_reg = H_fixed + epsilon * np.eye(H_fixed.shape[0])
    
    H_inv_fixed = np.linalg.inv(H_fixed_reg)
    
    for k in range(max_iter):
        g = grad(x)
        p = -H_inv_fixed @ g  # Направление спуска
        alpha = armijo(f, grad, x, p)

        x_new = x + alpha * p
        path.append(x_new.copy())

        # Критерий остановки по изменению точки
        if np.linalg.norm(x_new - x) < 0.001:
            print(f"Сошлось за {k} итераций")
            break

        x = x_new

    else:
        print("Достигнуто максимальное число итераций")

    return x, path


x0 = [0, 0]
x_min, path = newton_method(himmelblau, grad_himmelblau, hess_himmelblau, x0)

print("Найденный минимум:", x_min)
print("Значение функции в минимуме:", himmelblau(x_min))

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
plt.plot(x_min[0], x_min[1], 'go', markersize=8, label='Найденный минимум')

for xm, ym in minima:
    plt.plot(xm, ym, 'mx', markersize=10, label='Известный минимум')

plt.title('Минимизация функции Химмельблау модифицированным методом Ньютона')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()