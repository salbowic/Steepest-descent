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

# Objective functions
# Booth function
def q(x):
    return f4(x)
# Function 1 from CEC 2017
def q1(x):
    return f1(x)
# Function 2 from CEC 2017
def q2(x):
    return f2(x)
# Function 3 from CEC 2017
def q3(x):
    return f3(x)

def steepest_descent(x, q, final_diff, beta, B_param):
    '''
    Steepest descent method for minimization
    :param x: starting point
    :param q: objective function
    :param final_diff: minimum difference between consecutive gradients,
    below which the algorithm is stopped
    :param beta: Beta parameter
    :param B_param: parameter by which Beta is multiplied in each iteration
    of the algorithm
    '''
    prev_gradient = np.zeros_like(x)
    prev_beta = beta
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                # Calculate the gradient
                grad_fct = grad(q)
                gradient = -grad_fct(x)
                # Determine the difference between the new and the previous gradient
                gradient_difference = np.linalg.norm(gradient - prev_gradient)
                # End the algorithm when the determined difference is less than the set value
                if gradient_difference < final_diff:
                    break
                # Determine the next point x, multiply Beta by a set value, and save the gradient as "previous"
                prev_beta = beta
                beta = beta * B_param
                x = x + beta * gradient
                x = np.clip(x, -100, 100)
                
                prev_gradient = gradient
                # Draw the gradient vector
                plt.arrow(x[0] - beta * gradient[0], x[1] - beta * gradient[1], beta * gradient[0], beta * gradient[1], head_width=1.5, head_length=3, fc='k', ec='k', zorder=100)
            except RuntimeWarning as e:
                print("RuntimeWarning, attempting to find the solution with smaller accuracy")
                break
    return x    

# Results
def solution(X, Y, x, qx, final_diff, beta, B_param):
    '''
    Function that uses the steepest_descent method to return the solution and display a plot
    showing the gradient vectors against the background of the objective function
    :param X: X coordinates
    :param Y: Y coordinates
    :param x, q, qx, final_diff, beta, B_param: parameters used by the solution function
    '''
    Z = np.empty(X.shape)
    show_plot = True
    q_start = qx(x)
    x_best = steepest_descent(x, qx, final_diff, beta, B_param)
    q_final = qx(x_best)
    if q_final >= q_start:
        if show_plot:
            plt.clf()  # Clear the plot
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

        # Draw the plot
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

    # Randomize point x:
    x2 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
    x1 = x2[:2]

    print(Fore.BLUE, "1. Booth x: ", x1, Fore.RESET)
    print(Fore.BLUE, "2. CEC   x: ", x2, Fore.RESET)

    MAX_X = 100
    PLOT_STEP = 0.1
    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)

    # Parameters for the task: (X, Y, x1, qx, final_diff, beta, B_param)
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
