import numpy as np

def to_canon(c, A_le, b_le, A_eq, b_eq, A_ge, b_ge):
    # Шаг 1) Приводим к канонич. виду
    num_vars = len(c)
    num_le = len(b_le) if b_le is not None else 0
    num_ge = len(b_ge) if b_ge is not None else 0
    num_eq = len(b_eq) if b_eq is not None else 0

    total_vars = num_vars + num_le + num_ge
    A_canon = np.zeros((num_le + num_ge + num_eq, total_vars))
    b_canon = np.zeros(num_le + num_ge + num_eq)

    row = 0

    # <=
    if num_le > 0:
        for i in range(num_le):
            A_canon[row, :num_vars] = A_le[i]
            A_canon[row, num_vars + i] = 1
            b_canon[row] = b_le[i]
            row += 1

    # >=
    if num_ge > 0:
        for i in range(num_ge):
            A_canon[row, :num_vars] = A_ge[i]
            A_canon[row, num_vars + num_le + i] = -1
            b_canon[row] = b_ge[i]
            row += 1

    # =
    if num_eq > 0:
        for i in range(num_eq):
            A_canon[row, :num_vars] = A_eq[i]
            b_canon[row] = b_eq[i]
            row += 1

    #если b<0, умножаем строку на -1
    for i in range(len(b_canon)):
        if b_canon[i] < 0:
            A_canon[i, :] *= -1
            b_canon[i] *= -1

    # Проверяем b >= 0
    if np.any(b_canon < 0):
        raise ValueError("Правая часть должна быть b >= 0")

    # Целевая функц.
    c_canon = np.concatenate([c, np.zeros(num_le + num_ge)])
    # Все переменные предполагаются неотрицат. x1, x2, ..., xn >= 0
    return A_canon, b_canon, c_canon, num_vars


def vspomogat_task(A, b):
    # Шаг 2) Построение вспомогат задачи
    # Вводятся искусств. переменные для формирования начальн. базиса

    num_rows, num_cols = A.shape
    # Добавляем искусств. переменные для каждого уравнения
    A_aux = np.hstack([A, np.eye(num_rows)])

    # Целевая функция вспомогат. задачи (минимиз. суммы искусств. переменных)
    c_aux = np.concatenate([np.zeros(num_cols), np.ones(num_rows)])

    # Формируем симплекс-таблицу [A|b]
    tableau = np.zeros((num_rows + 1, num_cols + num_rows + 1))
    tableau[:-1, :-1] = A_aux
    tableau[:-1, -1] = b
    tableau[-1, :-1] = c_aux

    # Выражаем искусств. переменные через базис (обнуляем коэффы в строке Z)
    for i in range(num_rows):
        tableau[-1, :] -= tableau[i, :]

    basis = list(range(num_cols, num_cols + num_rows))
    return tableau, basis, num_cols


def simplex(tableau, basis):
    # Шаг 3 Итерация симплекс-метода
    # Возвращает False если достигнут оптимум

    #Ищем разрешающий столбец (наим отрицат. эл. в строке Z)
    z_row = tableau[-1, :-1]
    if np.min(z_row) >= -1e-9:
        return False  # Оптимум достигнут

    pivot_col = np.argmin(z_row)

    #Ищем разрешающую строку (мин положит отношение b/a)
    ratios = np.zeros(tableau.shape[0] - 1)
    for i in range(tableau.shape[0] - 1):
        element = tableau[i, pivot_col]
        if element > 1e-9:
            ratios[i] = tableau[i, -1] / element
        else:
            ratios[i] = np.inf

    if np.all(ratios == np.inf):
        raise ValueError("Область доп. решений не огранич. (функц.уходит в бесконечность)")

    pivot_row = np.argmin(ratios)

    # Пересчет таблицы (метод Гаусса-Жордана)
    pivot_element = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] /= pivot_element

    for i in range(tableau.shape[0]):
        if i != pivot_row:
            tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    basis[pivot_row] = pivot_col
    return True


def main_task(tableau, basis, original_c, num_original_cols):
    #Шаг 4 Переход к основной задаче
    # Удаляются искусств. переменные, восстанавл. исходн. целевая функция.

    # Проверка, что сущ. допустим. решение (вспомогательная функция равна 0)
    if abs(tableau[-1, -1]) > 1e-9:
        raise ValueError("Не имеет допустимых решений (область пуста)")

    # Удаляем столбцы искусств. переменных
    tableau = np.delete(tableau, np.s_[num_original_cols:-1], axis=1)

    #Восстанавл. строку исходной целевой функц.
    new_z_row = np.zeros(tableau.shape[1])
    new_z_row[:len(original_c)] = original_c
    tableau[-1, :] = new_z_row

    # Обнуляем. коэффы базисных переменных в новой строке Z
    for i in range(len(basis)):
        bas_var = basis[i]
        tableau[-1, :] -= tableau[-1, bas_var] * tableau[i, :]

    return tableau


def solve(c, A_le=None, b_le=None, A_eq=None, b_eq=None, A_ge=None, b_ge=None):
    # Шаг 1 Канонич. вид
    A, b, c_canon, orig_vars = to_canon(c, A_le, b_le, A_eq, b_eq, A_ge, b_ge)

    # Шаг 2 Создание вспомогат задачи
    tableau, basis, num_canon_vars = vspomogat_task(A, b)

    #Шаг 3 Решение вспомогат. задачи
    while simplex(tableau, basis):
        pass

    # Шаг 4 Переход к основной задаче
    tableau = main_task(tableau, basis, c_canon, num_canon_vars)

    #Шаг 5 Решение основной задачи
    while simplex(tableau, basis):
        pass

    #Ответ
    solution = np.zeros(orig_vars)
    for i in range(len(basis)):
        if basis[i] < orig_vars:
            solution[basis[i]] = tableau[i, -1]

    optimal_value = -tableau[-1, -1]  # Значение Z
    return solution, optimal_value


#Решение варианта 6
if __name__ == "__main__":
    # Целевая функция Z = 2x1+x2+x3+3x4 -> min
    c = np.array([2, 1, 1, 3])

    # Ограничение <= : x1+2x2+x4 <= 10
    A_le = np.array([[1, 2, 0, 1]])
    b_le = np.array([10])

    # Ограничение = : x1+x3+x4 = 7
    A_eq = np.array([[1, 0, 1, 1]])
    b_eq = np.array([7])

    # Ограничение >= : x2+2x3 >= 5
    A_ge = np.array([[0, 1, 2, 0]])
    b_ge = np.array([5])

    x_opt, z_min = solve(c, A_le, b_le, A_eq, b_eq, A_ge, b_ge)
    print(f"Опт. точка: {np.round(x_opt, 3)}")
    print(f"Мин. значение целевой функции: {np.round(z_min, 3)}")