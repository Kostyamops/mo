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
    # Шаг 2) Построение вспомогат. задачи
    num_rows, num_cols = A.shape

    # Ищем уже существующие базисные переменные
    basis = [-1] * num_rows

    for j in range(num_cols):
        column = A[:, j]

        # Проверяем, является ли столбец единичным
        for i in range(num_rows):
            if (
                abs(column[i] - 1) < 1e-9
                and np.isclose(np.sum(np.abs(column)), 1)
                and basis[i] == -1
            ):
                basis[i] = j
                break

    # Ищем строки без базисных переменных
    artificial_rows = []

    for i in range(num_rows):
        if basis[i] == -1:
            artificial_rows.append(i)

    # Добавляем искусств. переменные только там,
    # где нет базисной переменной
    A_aux = np.hstack([
        A,
        np.eye(num_rows)[:, artificial_rows]
    ])

    num_artificial = len(artificial_rows)

    # Целевая функц. вспомогат. задачи:
    # сумма искусств. переменных -> min
    c_aux = np.concatenate([
        np.zeros(num_cols),
        np.ones(num_artificial)
    ])

    # Формируем симплекс-таблицу [A|b]
    tableau = np.zeros((
        num_rows + 1,
        num_cols + num_artificial + 1
    ))

    tableau[:-1, :-1] = A_aux
    tableau[:-1, -1] = b
    tableau[-1, :-1] = c_aux

    # Добавляем искусств. переменные в базис
    for k, i in enumerate(artificial_rows):
        basis[i] = num_cols + k

    # Обнуляем коэффы базисных переменных в строке W'
    for i in range(num_rows):
        tableau[-1, :] -= tableau[-1, basis[i]] * tableau[i, :]

    # Номера столбцов искусств. переменных
    artificial_cols = list(
        range(num_cols, num_cols + num_artificial)
    )

    return tableau, basis, num_cols, artificial_cols


def simplex(tableau, basis):
    # Шаг 3) Итерация симплекс-метода

    z_row = tableau[-1, :-1]
    # Если отрицат. коэффов нет — минимум найден
    if np.min(z_row) >= -1e-9:
        return False

    # Разрешающий столбец — самый отрицат. элемент
    pivot_col = np.argmin(z_row)

    # Разрешающая строка — минимальное положит. b/a
    ratios = np.full(tableau.shape[0] - 1, np.inf)

    for i in range(tableau.shape[0] - 1):
        a = tableau[i, pivot_col]

        if a > 1e-9:
            ratios[i] = tableau[i, -1] / a

    # Если положит. элементов нет — функция не ограничена снизу
    if np.all(np.isinf(ratios)):
        raise ValueError("Целевая функция не ограничена снизу")

    pivot_row = np.argmin(ratios)

    # Пересчет таблицы
    pivot_element = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] /= pivot_element

    for i in range(tableau.shape[0]):
        if i != pivot_row:
            tableau[i, :] -= (
                tableau[i, pivot_col] *
                tableau[pivot_row, :]
            )

    # Меняем базис
    basis[pivot_row] = pivot_col
    return True


def main_task(tableau, basis, original_c, num_original_cols, artificial_cols):
    # Шаг 4) Переход к основной задаче

    # Проверяем, что вспомогат. задача дала допустимое решение
    if abs(tableau[-1, -1]) > 1e-9:
        raise ValueError("Не имеет допустимых решений")

    # Проверяем искусств. переменные в базисе
    i = 0
    while i < len(basis):
        bas_var = basis[i]

        if bas_var in artificial_cols:
            if abs(tableau[i, -1]) > 1e-9:
                raise ValueError("Не имеет допустимых решений")

            # Ищем обычную переменную для выхода искусств. переменной
            pivot_col = None

            for j in range(tableau.shape[1] - 1):
                if j not in artificial_cols and j not in basis:
                    if abs(tableau[i, j]) > 1e-9:
                        pivot_col = j
                        break

            # Если обычной переменной нет — удаляем строку
            if pivot_col is None:
                tableau = np.delete(tableau, i, axis=0)
                basis.pop(i)
                continue

            # Пересчет таблицы
            pivot_element = tableau[i, pivot_col]
            tableau[i, :] /= pivot_element

            for j in range(tableau.shape[0]):
                if j != i:
                    tableau[j, :] -= (
                        tableau[j, pivot_col] *
                        tableau[i, :]
                    )

            # Меняем базис
            basis[i] = pivot_col

        i += 1

    # Удаляем искусств. переменные
    tableau = np.delete(tableau, artificial_cols, axis=1)

    # Исправляем индексы базисных переменных
    new_basis = []

    for bas_var in basis:
        shift = sum(col < bas_var for col in artificial_cols)
        new_basis.append(bas_var - shift)

    basis[:] = new_basis

    # Восстанавл. исходную целевую функцию
    new_z_row = np.zeros(tableau.shape[1])
    new_z_row[:len(original_c)] = original_c
    tableau[-1, :] = new_z_row

    # Обнуляем коэффы базисных переменных в строке W
    for i in range(len(basis)):
        bas_var = basis[i]
        tableau[-1, :] -= tableau[-1, bas_var] * tableau[i, :]

    return tableau


def solve(c, A_le=None, b_le=None, A_eq=None, b_eq=None, A_ge=None, b_ge=None):
    # Шаг 1 Канонич. вид
    A, b, c_canon, orig_vars = to_canon(c, A_le, b_le, A_eq, b_eq, A_ge, b_ge)

    # Шаг 2 Создание вспомогат. задачи
    tableau, basis, num_canon_vars, artificial_cols = vspomogat_task(A, b)

    #Шаг 3 Решение вспомогат. задачи
    while simplex(tableau, basis):
        pass

    # Шаг 4 Переход к основной задаче
    tableau = main_task(
        tableau,
        basis,
        c_canon,
        num_canon_vars,
        artificial_cols
    )

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