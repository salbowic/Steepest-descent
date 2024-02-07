import warnings
import cec2017
import numpy as np
import matplotlib.pyplot as plt

from cec2017.functions import f1
from cec2017.functions import f2
from cec2017.functions import f3
from autograd import grad
from colorama import Fore

def f4(x):
    return (x[0] + 2*x[1] -7)**2 + (2 * x[0] + x[1] - 5)**2

#funkcje celu
#booth
def q(x):
    return f4(x)
#funkcja 1 z CEC 2017
def q1(x):
    return f1(x)
#funkcja 2 z CEC 2017
def q2(x):
    return f2(x)
#funkcja 3 z CEC 2017
def q3(x):
    return f3(x)

def steepest_descent(x, q, final_diff, beta, B_param):
    '''
    Metoda najszybszego spadku (steepest descent) dla minimaliacji
    :param x: punkt startowy
    :param q: funkcja celu
    :param final_diff: minimalna różnica pomiędzy kolejnymi wyliczanymi gradientami,
    poniżej której algorytm jest zatrzymywany
    :param beta: parametr Beta
    :param B_param: parametr przez który mnożony jest parametr Beta w każdej iteracji
    algorytmu
    '''
    prev_gradient = np.zeros_like(x)
    prev_beta = beta
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                #wyznacz gradient
                grad_fct = grad(q)
                gradient = -grad_fct(x)
                #wyznacz różnicę między nowym, a poprzednim gradientem
                gradient_difference = np.linalg.norm(gradient - prev_gradient)
                #zakończ działanie algorytmu kiedy wyznaczona różnica jest mniejsza niż ustalona wartośc
                if gradient_difference < final_diff:
                    break
                #wyznacz następny punkt x, przemnóż Betę przez ustaloną wartość i zapisz gradient jako "poprzedni"
                prev_beta = beta
                beta = beta * B_param
                x = x + beta * gradient
                x = np.clip(x, -100, 100)
                
                prev_gradient = gradient
                #narysuj wektor gradientu
                plt.arrow(x[0] - beta * gradient[0], x[1] - beta * gradient[1], beta * gradient[0], beta * gradient[1], head_width=1.5, head_length=3, fc='k', ec='k', zorder=100)
            except RuntimeWarning as e:
                print("RuntimeWarning, attempting to find the solution with smaller accuracy")
                break
    return x    

#Wyniki
def solution(X, Y, x, qx, final_diff, beta, B_param):
    '''
    Funkcja wykorzystująca metode steepest_descent do zwrócenia rozwiązania i wyświetlenia wykresu
    przedstawiającego wektory gradientów na tle funkcji celu
    :param X: Współrzędne X
    :param Y: Współrzędne Y
    :param x, q, qx, final_diff, beta, B_param : parametry wykorzystywane przez funkcję solution
    '''
    Z = np.empty(X.shape)
    show_plot = True
    q_start = qx(x)
    x_best = steepest_descent(x, qx, final_diff, beta, B_param)
    q_final = qx(x_best)
    if q_final >= q_start:
        if show_plot:
            plt.clf()  # Wyczyść wykres
            show_plot = False
        solution(X, Y, x, qx, final_diff * 10, beta, B_param)
    else:
        if qx == q:
            title = "Booth"
            print(Fore.GREEN + title + Fore.RESET)
        elif qx == q1:
            title = "f1 CEC 2017"
            print(Fore.GREEN + title + Fore.RESET)
        elif qx == q2:
            title = "f2 CEC 2017"
            print(Fore.GREEN + title + Fore.RESET)
        elif qx == q3:
            title = "f3 CEC 2017"
            print(Fore.GREEN + title + Fore.RESET)

        print('q(x) = %.6f' %(qx(x)))
        print("q(x*) = %.6f" % q_final)
        print("x* = ", x_best)

        #narysuj wykres
        if show_plot:    
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = qx(np.array([X[i, j], Y[i, j]]))
                    
            plt.contour(X, Y, Z, 40)
            plt.grid()
            plt.title(title)

def main():
    UPPER_BOUND = 100
    DIMENSIONALITY = 10

    #wylosuj punkt x:
    x2 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
    x1 = x2[:2]

    print(Fore.BLUE, "1. Booth x: ", x1, Fore.RESET)
    print(Fore.BLUE, "2. CEC   x: ", x2, Fore.RESET)

    MAX_X = 100
    PLOT_STEP = 0.1
    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)

    #parametry dla zadania: (X, Y, x1, qx, final_diff, beta, B_param)
    parameters = [
        (X, Y, x1, q, 1e-7, 0.04, 1),
        (X, Y, x2, q1, 1e+3, 8e-9, 1),
        (X, Y, x2, q2, 9e+5, 3e-20, 4),
        (X, Y, x2, q3, 1e+5, 4e-9, 1)
    ]

    for p in parameters:
        plt.figure()
        solution(*p)
    plt.show()

if __name__ == "__main__":
    main()