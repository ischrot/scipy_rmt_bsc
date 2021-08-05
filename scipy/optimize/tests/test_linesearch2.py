"""
Tests for line search routines
"""
from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
                           assert_array_almost_equal_nulp, assert_warns,
                           suppress_warnings)
import scipy.optimize.linesearch as ls
import scipy.optimize.nonlin as nl #(LS)
from scipy.linalg import norm
from scipy.optimize.linesearch import LineSearchWarning
import numpy as np
from copy import deepcopy  # (IS)


###(LS)###
def assert_rmt(alpha, dx, F0, Fx_new, jacobian, param, err_msg=""):
    """
    Check that RMT condition applies
    """


    #parameters = ls.prepare_parameters('rmt',param,jacobian,dx)
    #keep parameters from prepare_parameters in ls
    parameters = param
    rmt_eta_upper = parameters['rmt_eta_upper']
    rmt_eta_lower = parameters['rmt_eta_lower']
    amin = parameters['amin']

    #Step 1: Eval t_dx_omega
    dxbar = jacobian.solve(
                Fx_new
            )

    dx_diff = dxbar + (1 - alpha) * dx # note that dx = - J(x_k)^(-1)F(x_k)

    nominator = 2 * norm(dx_diff)
    denominator = alpha * norm(dx)

    t_dx_omega = nominator / denominator

    tester = (rmt_eta_lower <= t_dx_omega and t_dx_omega <= rmt_eta_upper) or (rmt_eta_lower > t_dx_omega and alpha == 1.0)

    msg = "s = %s; f(x_0) = %s; f(x_0+s*dx) = %s; %s" % (alpha, F0, Fx_new, err_msg)
    assert_(tester or (alpha<amin), msg)


def assert_bsc(alpha, x, dx, func, old_jacobian, param, err_msg):
    #parameters = ls.prepare_parameters('bsc',param, old_jacobian, dx)
    parameters=param
    H_lower = parameters['H_lower']
    H_upper = parameters['H_upper']
    amin = parameters['amin']

    x_new = x + alpha * dx
    Fx_new = func(x_new)
    jacobian = deepcopy(old_jacobian)
    jacobian.update(
        x_new.copy(),
        Fx_new
    )
    dx_next_it = -jacobian.solve(
        Fx_new
    )
    dx_diff = dx_next_it - dx
    H_prime = alpha * norm(dx_diff)

    tester = (H_lower <= H_prime and H_prime <= H_upper) or (H_lower > H_prime and alpha >= 1.0)

    msg = "s = %s; phi(0) = %s; phi(s) = %s; %s" % (alpha, func(x), Fx_new, err_msg)

    assert_(tester or (alpha<amin), msg)
###(LS)###


def assert_fp_equal(x, y, err_msg="", nulp=50):
    """Assert two arrays are equal, up to some floating-point rounding error"""
    try:
        assert_array_almost_equal_nulp(x, y, nulp)
    except AssertionError as e:
        raise AssertionError("%s\n%s" % (e, err_msg)) from e


class TestLineSearch(object):

    # -- Test functions

    def _line_func_1(self, x):
        self.fcount += 1
        f = np.array([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0,
            0.5 * (x[1] - x[0]) ** 3 + x[1]])
        df = np.array([[1 + 1.5 * (x[0] - x[1]) ** 2,
                       - 1.5 * (x[0] - x[1]) ** 2],
                     [-1.5 * (x[1] - x[0]) ** 2,
                     1 + 1.5 * (x[1] - x[0]) ** 2]])
        return f, df

    #def _line_func_2(self, x):
    #    self.fcount += 1
    #    f = np.dot(x, np.dot(self.A, x)) + 1
    #    df = np.dot(self.A + self.A.T, x)
    #    return f, df

    # --

    def setup_method(self):
        self.line_funcs = []
        self.N = 2
        self.fcount = 0

        def bind_index(func, idx):
            # Remember Python's closure semantics!
            return lambda *a, **kw: func(*a, **kw)[idx]

        for name in sorted(dir(self)):
            if name.startswith('_line_func_'):
                value = getattr(self, name)
                self.line_funcs.append(
                    (name, bind_index(value, 0), bind_index(value, 1)))

        np.random.seed(1234)
        self.A = np.random.randn(self.N, self.N)


    def line_iter(self):
        for name, f, fprime in self.line_funcs:
            k = 0
            while k < 9:
                #x = [-10.0, 10.0]
                x = np.random.randn(self.N)
                p = np.random.randn(self.N)
                grad = lambda x: [fprime(x)[0,0],fprime(x)[1,1]]
                
                if np.dot(p, grad(x)) >= 0:
                    # always pick a descent direction
                    continue
                k += 1
                old_fv = float(np.random.randn())
                yield name, f, fprime, x, p, old_fv

    

    ###(LS)###
    ##RMT not usefull for scalar functions, thus no need for test_scalar_search_rmt?

    def test_line_search_rmt(self):
        #There is at least 1 function R^20->R to be tested, but this leads to s=None
        not_full_step = 0
        counter =0 
        for name, f, fprime, x, p, old_f in self.line_iter():
            jac = lambda x: fprime(x)
            x0 = nl._as_inexact(x)
            func = lambda z: nl._as_inexact(f(nl._array_like(z, x0))).flatten()
            x = x0.flatten()
            jacobian = nl.asjacobian(jac)
            jacobian.setup(x.copy(), f(x), func)
            options = {'jacobian': jacobian, 'jac_tol': min(1e-03,1e-03*norm(f(x))), 'amin':1e-8}

            ### We need a special search direction otherwise we get problems in calculating omega_F
            Fx = func(x)
            dx = -jacobian.solve(Fx, tol=options['jac_tol'])
            
            ###Now test different values of rmt_eta
            tester = [0.5, 0.6, 0.7, 0.8, 0.9, None] #None to test the case without given eta
            for eta_test in tester:
                if eta_test != None:
                    options['rmt_eta'] = eta_test
                s, dxbar, f_new = ls.scalar_search_rmt(func, x, dx, parameters=options)
                #here stepsize s is often (always??) equal to 1 due to fullfilling rmt_eta_lower > t_dx_omega and alpha == 1.0 (rmt_func)
                #is this an expected step size or is this wrong?
                if s!=1:
                    not_full_step += 1
                counter += 1
                if s == None:
                    s = 1
                assert_fp_equal(f_new, f(x+s*dx), name)
                assert_rmt(s, dx, f(x), f_new, jacobian, options, err_msg="%s" % name)
        print(f"{not_full_step} of {counter} test steps didn't lead to a full step")


    def test_line_search_bsc(self):
        #There is at least 1 function R^20->R to be tested, but this leads to s=None
        for name, f, fprime, x, p, old_f in self.line_iter():
            jac = lambda x: fprime(x)
            x0 = nl._as_inexact(x)
            func = lambda z: nl._as_inexact(f(nl._array_like(z, x0))).flatten()
            x = x0.flatten()
            jacobian = nl.asjacobian(jac)
            jacobian.setup(x.copy(), f(x), func)
            options = {'jacobian': jacobian, 'jac_tol': min(1e-03,1e-03*norm(f(x))), 'amin':1e-8}
            Fx = func(x)
            dx = -jacobian.solve(Fx, tol=options['jac_tol'])

            ### check with ENM step as for rmt
            s, f_new= ls.scalar_search_bsc(func, x, dx, Fx, parameters=options)
            
            assert_fp_equal(f_new, f(x+s*dx), name)
            assert_bsc(s, x, dx, func, jacobian, options, err_msg="%s" % name)

            ### check different descent direction (not ENM)
            s, f_new= ls.scalar_search_bsc(func, x, p, Fx, parameters=options)

            assert_fp_equal(f_new, f(x+s*p), name)
            assert_bsc(s, x, p, func, jacobian, options, err_msg="%s" % name)


#####TODO
#Another test of rmt with different rmt_eta <= 1.0 and starting values to get a situation in which
#rmt won't give back just the full step.