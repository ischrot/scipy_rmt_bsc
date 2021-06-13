#from scipy.optimize import root, show_options, rosen_der
from scipy import optimize
from scipy.optimize.tests import test_nonlin, test_linesearch, test__root, test_linesearch2
import scipy as sp

import numpy as np
import matplotlib.pyplot as plt
import time
#import pycutest

# probs = pycutest.find_problems(constraints='U', userN=True)
# print(len(probs))
# probs = pycutest.find_problems(constraints='U', userN=False)
# print(len(probs))
# probs = pycutest.find_problems(constraints='U')
# print(len(probs))

## R^2->R^2
def fun(x):
    return [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0,
            0.5 * (x[1] - x[0]) ** 3 + x[1]]
def jac(x):
    return np.array([[1 + 1.5 * (x[0] - x[1]) ** 2,
                       - 1.5 * (x[0] - x[1]) ** 2],
                     [-1.5 * (x[1] - x[0]) ** 2,
                     1 + 1.5 * (x[1] - x[0]) ** 2]])

##scalar function R->R
def fun2(x):
    return x[0]**3-np.cos(x[0])*np.sin(x[0])**2
def jac2(x):
    return np.sin(x[0])**3 - 2*np.cos(x[0])**2 *np.sin(x[0])+3*x**2

##R^n->R
def line_func(x):
    return np.dot(x, x)
def line_jac(x):
    return 2*x


def fun_jac(x):
   return fun(x), jac(x)

#
# def jac(x):
#    return np.array([[1.0, 0], [0, 1.0]])

line_search_options = {
    'rmt_eta': 10.0,
    'rmt_eta_lower': 5.0,
    'rmt_eta_upper': 20.2,
    'prediction': False,
    'amin': 1e-12,
    'back_projection': False,
    'ideal_projection': False,
    'analysis': False,
    # 'H': 0.01,
    # 'H_lower': 0.8, #0.2,
    'H_lower': lambda H: H * min(.1, H), #0.2,
    # 'H_upper': 1.2,
    'H_upper': lambda H: H * 2.0,
    'H_rel': 0.1,
    'smooth_factor': 0.8,
    'astall': 1e-10,
    'adaptive_int': False,
    'second_order': False,
    'min_search': False,
    'omega_mod': False,
}

jac_options = {
    'method': 'minres',
    'inner_maxiter': 100,
    'inner_tol': 1e-02
    # 'inner_atol': 1e-02
}

solver_options = {
    'line_search': 'rmt',
    'disp': True,
    'parameters': line_search_options,
#    'jac_options': jac_options,
    'xtol': 1e-3,
    'maxiter': 1000
}

# prob = pycutest.import_problem('OSBORNEA')
# prob = pycutest.import_problem('ROSENBR')
# x0 = [-10.0, 10.0]
# x0 = prob.x0
# x0 = [0.84, 0.15]
# f = prob.lagjac
# f = lambda x: prob.lagjac(x)[0]
# j = prob.hess

# print(prob)

# x = x0
f = fun
j = jac
x0 = [10.0,-10.0]
#sol = optimize.root(f, x0, jac=j, method='broyden1', options=solver_options)

# param={'line_search' : 'rmt'}

#start = time.time()
#sol = root(f, x0, jac=j, method='exact', options=solver_options)

#sol = root(f, x0, method='krylov', options=param)

#print(time.time() - start)
#time.sleep(60.0)
# sol = root(fun2, [2.0], method='exact', jac=jac2, options=solver_options)
# sol = root(fun_jac, x0, jac=True, method='exact', options=solver_options)
# show_options(solver='root')
# print("F_evals: ", line_search_options['F_evals'], "\nJ_evals:", line_search_options['J_evals']
#       , "\nJ_updates:", line_search_options['J_updates'])
# print(prob.report())
#print(sol)
"""
s = line_search_options['step_sizes']

dx = np.array(line_search_options['dx'])
x = np.array(line_search_options['x'])


plt.figure()
plt.plot(x[:, 0], x[:, 1], '-', marker='.')
plt.show()

plt.figure()
plt.plot(s, marker='.')
plt.yscale('log')
plt.show()

X = np.arange(min(x[:,0])-1.0, max(x[:,0])+1.0, 0.5)
nx = len(X)
Y = np.arange(min(x[:,1])-1.0, max(x[:,1])+1.0, 0.5)
ny = len(Y)
X, Y = np.meshgrid(X, Y, indexing='xy')
Z = np.zeros(X.shape)
for i in range(nx):
    for j in range(ny):
        # treat xv[j,i], yv[j,i]
        result = f(np.array([X[j,i],Y[j,i]]))
        Z[j, i] = np.linalg.norm(result)


# Plot the surface.
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')

zline = [np.linalg.norm(f(x[i, :])) for i in range(len(x))]

surf = ax.plot_surface(X, Y, np.array(Z)/np.max(Z), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha = 0.4)
ax.plot3D(x[:, 0], x[:, 1], np.array(zline)/np.max(Z), 'red', marker='x', markersize=8, label='Newton Iterates with RMT Linesearch')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Objective Value')
plt.title('Minimization of the Rosenbrock Function')
plt.legend(loc=(0.2, 0.75))

plt.savefig("NLO_picture.pdf")
plt.show()
"""
"""
###Test-Area###

parameters = ({},{'line_search': 'rmt'},{'line_search': 'rmt_int'},{'line_search': 'bsc'})
#hybr and lm are both not using any line_search, 'df-sane' uses different type of linesearch
methods = ('broyden1', 'broyden2','anderson','diagbroyden','excitingmixing','krylov','linearmixing')

evaluater1 = np.zeros((7,4))
evaluater2 = np.zeros((7,4))
for k in range(100):
    rand=np.random.uniform(-100,100,(1,2))
    x0=[rand[0,0],rand[0,1]]
    for j in range(len(parameters)):
        param=parameters[j]
        for i in range(len(methods)):
            meth=methods[i]
            #print(f"try {meth}")
            try:
                sol = root(f, x0, method=meth, options=param)
                #print(sol)
                if sol.success == True:
                    evaluater1[i,j]+=1
                else:
                    evaluater2[i,j]+=1
            except:
                #print(f"{meth} did not work")
                pass
print(f"Sucessfull runs:\n{evaluater1}")
print(f"Did not converge:\n{evaluater2}")
#results -> test_tabular.pdf



### UNIT-Test Area ###

#test__root
RootTester = test__root.TestRoot()
RootTester.test_tol_parameter()
#The following two don't use 'exact' method but are stil used to garantee general functionality
RootTester.test_minimize_scalar_coerce_args_param()
RootTester.test_f_size()
"""
#test_linesearch
LSTest = test_linesearch2.TestLineSearch()
LSTest.setup_method()

LSTest.test_line_search_rmt()
LSTest.test_line_search_bsc()






#
# optimize.show_options(solver='root',method='broyden1')
#TODO: Write proper evaluation script.
#TODO: Test Newton fractal

#print(sp.__version__)
#TODO: Implement A2-condition check as callback fct.(?)
#TODO: Plitts Hessian init for the approx. Jacobians (?)
#TODO: Update Changelog.txt
