"""
Functions
---------
.. autosummary::
   :toctree: generated/

    line_search_armijo
    line_search_wolfe1
    line_search_wolfe2
    scalar_search_wolfe1
    scalar_search_wolfe2

"""
from warnings import warn

from scipy.optimize import minpack2, root_scalar # (IS)
from scipy.linalg import norm #(IS)
from copy import deepcopy #(IS)
import numpy as np

__all__ = ['LineSearchWarning', 'line_search_wolfe1', 'line_search_wolfe2',
           'scalar_search_wolfe1', 'scalar_search_wolfe2',
           'line_search_armijo', 'scalar_search_rmt', 'scalar_search_bsc'] #(IS)

class LineSearchWarning(RuntimeWarning):
    pass


#------------------------------------------------------------------------------
# Minpack's Wolfe line and scalar searches
#------------------------------------------------------------------------------

def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction

    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`

    The rest of the parameters are the same as for `scalar_search_wolfe1`.

    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point

    """
    if gfk is None:
        gfk = fprime(xk)

    if isinstance(fprime, tuple):
        eps = fprime[1]
        fprime = fprime[0]
        newargs = (f, eps) + args
        gradient = False
    else:
        newargs = args
        gradient = True

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s*pk, *newargs)
        if gradient:
            gc[0] += 1
        else:
            fc[0] += len(xk) + 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
            phi, derphi, old_fval, old_old_fval, derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.

    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`

    Notes
    -----
    Uses routine DCSRCH from MINPACK.

    """

    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    maxiter = 100
    for i in range(maxiter):
        stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1,
                                                   c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        if task[:2] == b'FG':
            alpha1 = stp
            phi1 = phi(stp)
            derphi1 = derphi(stp)
        else:
            break
    else:
        # maxiter reached, the line search did not converge
        stp = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        stp = None  # failed

    return stp, phi1, phi0


line_search = line_search_wolfe1


#------------------------------------------------------------------------------
# Pure-Python Wolfe line and scalar searches
#------------------------------------------------------------------------------

def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
                       extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.

    Examples
    --------
    >>> from scipy.optimize import line_search

    A objective function and its gradient are defined.

    >>> def obj_func(x):
    ...     return (x[0])**2+(x[1])**2
    >>> def obj_grad(x):
    ...     return [2*x[0], 2*x[1]]

    We can find alpha that satisfies strong Wolfe conditions.

    >>> start_point = np.array([1.8, 1.7])
    >>> search_gradient = np.array([-1.0, -1.0])
    >>> line_search(obj_func, obj_grad, start_point, search_gradient)
    (1.0, 2, 1, 1.1300000000000001, 6.13, [1.6, 1.4])

    """
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha * pk, *args)

    if isinstance(myfprime, tuple):
        def derphi(alpha):
            fc[0] += len(xk) + 1
            eps = myfprime[1]
            fprime = myfprime[0]
            newargs = (f, eps) + args
            gval[0] = fprime(xk + alpha * pk, *newargs)  # store for later use
            gval_alpha[0] = alpha
            return np.dot(gval[0], pk)
    else:
        fprime = myfprime

        def derphi(alpha):
            gc[0] += 1
            gval[0] = fprime(xk + alpha * pk, *args)  # store for later use
            gval_alpha[0] = alpha
            return np.dot(gval[0], pk)

    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = np.dot(gfk, pk)

    if extra_condition is not None:
        # Add the current gradient as argument, to avoid needless
        # re-evaluation
        def extra_condition2(alpha, phi):
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha * pk
            return extra_condition(alpha, x, phi, gval[0])
    else:
        extra_condition2 = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
            extra_condition2, maxiter=maxiter)

    if derphi_star is None:
        warn('The line search algorithm did not converge', LineSearchWarning)
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star


def scalar_search_wolfe2(phi, derphi, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=None,
                         extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0.
    old_phi0 : float, optional
        Value of phi at previous point.
    derphi0 : float, optional
        Value of derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, phi_value)``
        returning a boolean. The line search accepts the value
        of ``alpha`` only if this callable returns ``True``.
        If the callable returns ``False`` for the step length,
        the algorithm will continue with new iterates.
        The callable is only called for iterates satisfying
        the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star.
    phi0 : float
        phi at 0.
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.

    """

    if phi0 is None:
        phi0 = phi(0.)

    if derphi0 is None:
        derphi0 = derphi(0.)

    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)
    #derphi_a1 = derphi(alpha1) evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    if extra_condition is None:
        extra_condition = lambda alpha, phi: True

    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None

            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = "The line search algorithm could not find a solution " + \
                      "less than or equal to amax: %s" % amax

            warn(msg, LineSearchWarning)
            break

        not_first_iteration = i > 0
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and not_first_iteration):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warn('The line search algorithm did not converge', LineSearchWarning)

    return alpha_star, phi_star, phi0, derphi_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found, return None.

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
    
    Part of the optimization algorithm in `scalar_search_wolfe2`.
    
    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.

    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


#------------------------------------------------------------------------------
# Armijo line and scalar searches
#------------------------------------------------------------------------------

def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Parameters
    ----------
    f : callable
        Function to be minimized.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    gfk : array_like
        Gradient of `f` at point `xk`.
    old_fval : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Optional arguments.
    c1 : float, optional
        Value to control stopping criterion.
    alpha0 : scalar, optional
        Value of `alpha` at start of the optimization.

    Returns
    -------
    alpha
    f_count
    f_val_at_alpha

    Notes
    -----
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1*pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    derphi0 = np.dot(gfk, pk)
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                       alpha0=alpha0)
    return alpha, fc[0], phi1


def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
    """
    Compatibility wrapper for `line_search_armijo`
    """
    r = line_search_armijo(f, xk, pk, gfk, old_fval, args=args, c1=c1,
                           alpha0=alpha0)
    return r[0], r[1], 0, r[2]


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """Minimize over alpha, the function ``phi(alpha)``.

    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    alpha > 0 is assumed to be a descent direction.

    Returns
    -------
    alpha
    phi1

    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1


#------------------------------------------------------------------------------
# Non-monotone line search for DF-SANE
#------------------------------------------------------------------------------

def _nonmonotone_line_search_cruz(f, x_k, d, prev_fs, eta,
                                  gamma=1e-4, tau_min=0.1, tau_max=0.5):
    """
    Nonmonotone backtracking line search as described in [1]_

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    prev_fs : float
        List of previous merit function values. Should have ``len(prev_fs) <= M``
        where ``M`` is the nonmonotonicity window parameter.
    eta : float
        Allowed merit function increase, see [1]_
    gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position

    References
    ----------
    [1] "Spectral residual method without gradient information for solving
        large-scale nonlinear systems of equations." W. La Cruz,
        J.M. Martinez, M. Raydan. Math. Comp. **75**, 1429 (2006).

    """
    f_k = prev_fs[-1]
    f_bar = max(prev_fs)

    alpha_p = 1
    alpha_m = 1
    alpha = 1

    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)

        if fp <= f_bar + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        xp = x_k - alpha_m * d
        fp, Fp = f(xp)

        if fp <= f_bar + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    return alpha, xp, fp, Fp


def _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta,
                                   gamma=1e-4, tau_min=0.1, tau_max=0.5,
                                   nu=0.85):
    """
    Nonmonotone line search from [1]

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    f_k : float
        Initial merit function value.
    C, Q : float
        Control parameters. On the first iteration, give values
        Q=1.0, C=f_k
    eta : float
        Allowed merit function increase, see [1]_
    nu, gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position
    C : float
        New value for the control parameter C
    Q : float
        New value for the control parameter Q

    References
    ----------
    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line
           search and its application to the spectral residual
           method'', IMA J. Numer. Anal. 29, 814 (2009).

    """
    alpha_p = 1
    alpha_m = 1
    alpha = 1

    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)

        if fp <= C + eta - gamma * alpha_p**2 * f_k:
            alpha = alpha_p
            break

        alpha_tp = alpha_p**2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        xp = x_k - alpha_m * d
        fp, Fp = f(xp)

        if fp <= C + eta - gamma * alpha_m**2 * f_k:
            alpha = -alpha_m
            break

        alpha_tm = alpha_m**2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)

    # Update C and Q
    Q_next = nu * Q + 1
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next

    return alpha, xp, fp, Fp, C, Q



##(LS)

def prepare_parameters(line_search, parameters, jacobian, dx):

    #parameters['jacobian'] = jacobian  # (LS) already done when using prepare_parameters in solution 2. For 1 we would need it

    if line_search in ('rmt', 'rmt_int'):
        if 'rmt_eta' not in parameters:
            warn("Necessary line search option 'rmt_eta' not specified."
                 " Using default (1.0).", LineSearchWarning)
            parameters['rmt_eta'] = 1.0

        elif parameters['rmt_eta'] <= 0.0:
        # elif parameters['rmt_eta'] >= 2.0 or parameters['rmt_eta'] <= 0.0:
            warn("Line search option 'rmt_eta' is negative."
                 " Using default (1.0) instead.", LineSearchWarning)
            # warn("Line search option 'rmt_eta' exceeds its maximum (2.0) or is negative."
            #      " Using default (1.0) instead.", LineSearchWarning)
            parameters['rmt_eta'] = 1.0

    if line_search in ('bsc', 'rmt'):
        if 'min_search' not in parameters:
            warn("Necessary line search option 'min_search' not specified."
                 " Using default (False).", LineSearchWarning)
            parameters['min_search'] = False
        elif parameters['min_search'] not in (True, False):
            warn("Necessary line search option 'min_search' not specified correctly."
                 " Using default (False).", LineSearchWarning)
            parameters['min_search'] = False

        if 'nr_intervals' not in parameters:
            if parameters['min_search']:
                warn("Necessary line search option 'nr_intervals' not specified."
                     " Using default (20).", LineSearchWarning)
            parameters['nr_intervals'] = 20
        elif type(parameters['nr_intervals']).__name__ != 'int' or parameters['nr_intervals'] < 1:
            warn("Necessary line search option 'nr_intervals' not specified correctly."
                 " Using default (20).", LineSearchWarning)
            parameters['nr_intervals'] = 20

    if line_search == 'rmt_int':
        if 'adaptive_int' not in parameters:
            warn("Necessary line search option 'adaptive_int' not specified."
                 " Using default (False).", LineSearchWarning)
            parameters['adaptive_int'] = False
        elif parameters['adaptive_int'] not in (True, False):
            warn("Necessary line search option 'adaptive_int' not specified correctly."
                 " Using default (False).", LineSearchWarning)
            parameters['adaptive_int'] = False
        if 'second_order' not in parameters:
            warn("Necessary line search option 'second_order' not specified."
                 " Using default (False).", LineSearchWarning)
            parameters['second_order'] = False
        elif parameters['second_order'] not in (True, False):
            warn("Necessary line search option 'second_order' not specified correctly."
                 " Using default (False).", LineSearchWarning)
            parameters['second_order'] = False
        if 'test_step' not in parameters:
            warn("Necessary line search option 'test_step' not specified."
                 " Using default (0.1).", LineSearchWarning)
            parameters['test_step'] = 0.1
        elif parameters['test_step'] <= 0.0 or parameters['test_step'] >= 1.0:
            warn("Necessary line search option 'test_step' not specified correctly."
                 " Using default (0.1).", LineSearchWarning)
            parameters['test_step'] = 0.1

    if line_search == 'rmt':
        if 'rmt_eta_upper' not in parameters:
            warn("Necessary line search option 'rmt_eta_upper' not specified."
                 " Using default (1.2).", LineSearchWarning)
            parameters['rmt_eta_upper'] = 1.2

        elif parameters['rmt_eta_upper'] <= parameters['rmt_eta']:
            warn("rmt_eta_upper must be bigger than 'rmt_eta'. Using default (1.2 * rmt_eta)", LineSearchWarning)
            parameters['rmt_eta_upper'] = 1.2 * parameters['rmt_eta']

        # (IS): Disabled this check to test RMT with higher rmt_upper
        # if parameters['rmt_eta_upper'] >= 2.0:
        #     warn("rmt_eta * rmt_eta_upper exceeds its maximum (2.0)."
        #          " Using default (rmt_eta=1.0, rmt_eta_upper=1.2, "
        #          "rmt_eta_lower=0.8)", LineSearchWarning)
        #     parameters['rmt_eta_upper'] = 1.2
        #     parameters['rmt_eta'] = 1.0
        #     parameters['rmt_eta_lower'] = 0.8

        if 'rmt_eta_lower' not in parameters:
            warn("Necessary line search option 'rmt_eta_lower' not specified."
                 " Using default (0.8 * rmt_eta).", LineSearchWarning)
            parameters['rmt_eta_lower'] = 0.8 * parameters['rmt_eta']

        elif parameters['rmt_eta_lower'] < 0.0 or parameters['rmt_eta_lower'] >= parameters['rmt_eta']:
            warn("rmt_eta_lower is negative or not smaller than 'rmt_eta'."
                 " Using default (0.8 * rmt_eta)", LineSearchWarning)
            parameters['rmt_eta_lower'] = 0.8 * parameters['rmt_eta']

        # parameters['rmt_eta_upper'] *= parameters['rmt_eta']
        # parameters['rmt_eta_lower'] *= parameters['rmt_eta']

        if 'prediction' not in parameters:
            parameters['prediction'] = True

        if 'amin' not in parameters or parameters['amin'] < 0.0 \
                or parameters['amin'] > 1.0:
            warn("Necessary line search option 'amin' not (correctly) specified."
                 " Using default (1e-8).", LineSearchWarning)
            parameters['amin'] = 1e-8

        if 'astall' not in parameters or parameters['astall'] < 0.0 \
                or parameters['astall'] > 1.0:
            warn("Necessary line search option 'astall' not (correctly) specified."
                 " Using default (1e-10).", LineSearchWarning)
            parameters['astall'] = 1e-10

        if 'back_projection' not in parameters:
            parameters['back_projection'] = False

        if 'ideal_projection' not in parameters:
            parameters['ideal_projection'] = False

        if parameters['ideal_projection'] and not parameters['back_projection']:
            parameters['back_projection'] = True

        if 'omega_mod' not in parameters:
            warn("Necessary line search option 'omega_mod' not specified."
                 " Using default (False).", LineSearchWarning)
            parameters['omega_mod'] = False
        elif parameters['omega_mod'] not in (True, False):
            warn("Necessary line search option 'omega_mod' not specified correctly."
                 " Using default (False).", LineSearchWarning)
            parameters['omega_mod'] = False

    elif line_search == 'bsc':
        if 'H' not in parameters and 'H_rel' not in parameters:
            warn("Necessary line search option 'H' or 'H_rel' not specified."
                 " Using default (H_rel = 1.0).", LineSearchWarning)
            parameters['H_rel'] = 1.0
            parameters['H'] = parameters['H_rel'] * max(1.0, norm(dx))
        elif 'H' in parameters and 'H_rel' in parameters:
            if parameters['H'] > 0.0 and parameters['H_rel'] > 0.0:
                warn("Both line search options 'H' or 'H_rel' specified."
                     " Using 'H_rel' by default.", LineSearchWarning)
                parameters['H'] = parameters['H_rel'] * max(1.0, norm(dx))
            elif parameters['H'] <= 0.0 and parameters['H_rel'] <= 0.0:
                warn("Both line search options 'H' or 'H_rel' are not specified (correctly)."
                     " Using default (H_rel=1.0).", LineSearchWarning)
                parameters['H_rel'] = 1.0
                parameters['H'] = parameters['H_rel'] * max(1.0, norm(dx))
            elif parameters['H_rel'] > 0.0:
                warn("Line search option 'H' is negative."
                     " Using correctly given 'H_rel'.", LineSearchWarning)
                parameters['H'] = parameters['H_rel'] * max(1.0, norm(dx))
        elif 'H' in parameters:
            parameters['H_rel'] = None
            if parameters['H'] <= 0.0:
                warn("Line search option 'H' is negative."
                     " Using default (2.0).", LineSearchWarning)
                parameters['H'] = 2.0
        elif 'H_rel' in parameters:
            if parameters['H_rel'] <= 0.0:
                warn("Line search option 'H_rel' is negative."
                     " Using default (1.0).", LineSearchWarning)
                parameters['H_rel'] = 1.0
            parameters['H'] = parameters['H_rel'] * max(1.0, norm(dx))

        if 'H_lower' not in parameters:
            warn("Necessary line search option 'H_lower' not specified."
                 " Using default (H*min(0.1, H)).", LineSearchWarning)
            parameters['H_lower'] = parameters['H'] * min(0.1, parameters['H'])
        elif callable(parameters['H_lower']):
            parameters['H_lower'] = parameters['H_lower'](parameters['H'])

        if parameters['H_lower'] < 0.0 or parameters['H_lower'] >= parameters['H']:
            warn("Line search option 'H_lower' is negative or bigger than 'H'."
                 " Using default (H*min(0.1, H)).", LineSearchWarning)
            parameters['H_lower'] = parameters['H'] * min(0.1, parameters['H'])

        if 'H_upper' not in parameters:
            warn("Necessary line search option 'H_upper' not specified."
                 " Using default (2*H).", LineSearchWarning)
            parameters['H_upper'] = 2.0 * parameters['H']
        elif callable(parameters['H_upper']):
            parameters['H_upper'] = parameters['H_upper'](parameters['H'])
        if parameters['H_upper'] <= parameters['H']:
            warn("Line search option 'H_upper' is smaller than 'H'."
                 " Using default (2*H).", LineSearchWarning)
            parameters['H_upper'] = 2.0 * parameters['H']

        if 'smooth_factor' not in parameters or parameters['smooth_factor'] < 0.0 \
                or parameters['smooth_factor'] > 1.0:
            warn("Necessary line search option 'smooth_factor' not (correctly) specified."
                 " Using default (0.8).", LineSearchWarning)
            parameters['smooth_factor'] = 0.8

        if 'amin' not in parameters or parameters['amin'] < 0.0 \
                or parameters['amin'] > 1.0:
            warn("Necessary line search option 'amin' not (correctly) specified."
                 " Using default (1e-8).", LineSearchWarning)
            parameters['amin'] = 1e-8

        # if 'afull' not in parameters or parameters['afull'] <= 0.0 \
        #         or parameters['afull'] > 1.0:
        #     warn("Necessary line search option 'afull' not (correctly) specified."
        #          " Using default (0.999).", LineSearchWarning)
        #     parameters['afull'] = 0.999

        if 'astall' not in parameters or parameters['astall'] < 0.0 \
                or parameters['astall'] > 1.0:
            warn("Necessary line search option 'astall' not (correctly) specified."
                 " Using default (1e-10).", LineSearchWarning)
            parameters['astall'] = 1e-10

    if 'analysis' in parameters:
        if parameters['analysis']:
            parameters['step_sizes'] = []  # store step sizes
            parameters['dx'] = []  # store step directions as originally calculated
            parameters['x'] = []  # iterates
            if line_search in ('rmt', 'bsc', 'rmt_int'):
                parameters['F_evals'] = 1 # when this fct is called, we already evaluated Fx and calculated dx, but for the latter we cached Fx
                parameters['J_evals'] = 1 # see comment above
                parameters['J_updates'] = 1 # see comment above
            if line_search == 'rmt' and parameters['back_projection']:
                parameters['dx_project'] = [] # store finally taken step directions as given by the back projection
                parameters['ss_project'] = [] # original step size used internally in the projection
    else:
        parameters['analysis'] = False
### (LS)

### (IS)
#------------------------------------------------------------------------------
# Restrictive monotonicity test based line search
#------------------------------------------------------------------------------
def scalar_search_rmt(func, x, dx, parameters = None):
    """
    Restrictive monotonicity test based line search as described in [1]_

    Parameters
    ----------
    func : callable
        Function returning J(x_k)^-1 F(x_k + alpha dx)
    x : ndarray
        Current iterate
    dx : ndarray
        Search direction
    Fx : ndarray:
        F(x_k)
    parameters : dict
        Can contain the following parameters:
        i)      rmt_eta : float
                    necessary tuning parameter (default 0.8)
        ii)     old_omega : float
                    Final curvature estimate of the previous iteration
        iii)    pred_init_frame : ndarray
                    If a prediction for the step size is made, this defines the brackets for the
                    root finding procedure, which then is [left_factor * prediction, right_factor * prediction]
                    (default: [0.9, 1.1])
        iv)     pred_frame_growth : float
                    Factor for the increase of the bracket size for the prediction of the step size, if the
                    previous bracket did not contain a solution (default: 1.2)
        v)      maxiter : int
                    Maximum number of calls to scipys scalar rootfinding procedure (default: 10)
        vi)     prediction : bool
                    Should the prediction for the step size be made? (default: True)
        vii)    amin : float
                    Minimum step size (default: 1e-6)

    Returns
    -------
    alpha : float
        Step length

    References
    ----------
    [1] Bock H.G., Kostina E., Schlöder J.P. (2000) "On the Role of Natural
        Level Functions to Achieve Global Convergence for Damped Newton
        Methods." In: Powell M.J.D., Scholtes S. (eds) System Modelling and
        Optimization. CSMO 1999. IFIP — The International Federation for
        Information Processing, vol 46. Springer, Boston, MA
    """

    # =================================================================
    ###################################################################
    # read and prepare parameters and variables:
    ###################################################################
    # =================================================================
    
    #(LS)
    if parameters == None:
        print("Missing jacobian")
    prepare_parameters('rmt',parameters,parameters['jacobian'],dx)
    #(LS)
    rmt_eta = parameters['rmt_eta']
    rmt_eta_upper = parameters['rmt_eta_upper']
    rmt_eta_lower = parameters['rmt_eta_lower']

    amin = parameters['amin']
    astall = parameters['astall']

    prediction = parameters['prediction']

    # omega_F will store omega_F to make a prediction possible in the next iteration
    omega_F = [-1.0]
    dxbar_cache = [None]
    Fx_cache = [None]

    #flag whether t_dx_omega was already computed for one of the boundaries of the bracket
    predicted_t_dx_omega = [False]

    jacobian = parameters['jacobian']
    jac_tol = parameters['jac_tol']


    # =================================================================
    ###################################################################
    # define auxiliary functions:
    ###################################################################
    # =================================================================

    ###################################################################
    # function to calculate t_dx_omega:
    ###################################################################
    def eval_t_dx_omega(alpha, omega_F, dxbar_cache, Fx_cache):
        Fx_new = func(x + alpha * dx)
        if np.isnan(Fx_new).any() or np.isinf(Fx_new).any():
            return rmt_eta_upper + 1.0
        else:
            dxbar = jacobian.solve(
                Fx_new,
                tol=jac_tol
            )

            dx_diff = dxbar + (1 - alpha) * dx # note that dx = - J(x_k)^(-1)F(x_k)

            nominator = 2 * norm(dx_diff)
            denominator = alpha * norm(dx)

            Fx_cache[0] = Fx_new
            if prediction:
                omega_F[0] = nominator / (denominator**2) #store omega_F for next iteration
            if back_projection and not ideal_projection:
                dxbar_cache[0] = dxbar

            return nominator / denominator

    ###################################################################
    # function checking whether the RMT is satisfied:
    ###################################################################
    # to make it callable by root_scalar, we construct it such that
    # i) for all step sizes satisfying
    #   eta_lower <= t_k omega_F norm(dx) <= eta_upper
    #   we return 0.0, which aborts the root finding procedure and classifies
    #   the step size as feasible
    # ii) if the left inequality is violated, we return -1.0
    # iii) if the right inequality is violated, we return 1.0
    # This is necessary to be able to have different signs for the left and right end
    # of the interval in the bisection procedure.
    def rmt_fct(alpha, omega_F, predicted_t_dx_omega, dxbar_cache, Fx_cache, t_dx_omega=None):
        # although we call a root finding procedure, we only want to achieve
        # rmt_eta_upper >= t dx omega >= rmt_eta_lower. Hence, we set a negative result to 0.0 to end the root
        # finding procedure.
        # to avoid that left and right interval boundary have the same sign, we
        # return -1 for too small values.
        if alpha < amin:
            if predicted_t_dx_omega[0]:
                predicted_t_dx_omega[0] = False
            return -1.0

        else:
            #if we already calculated t_dx_omega, reuse it
            if not predicted_t_dx_omega[0]:
                if t_dx_omega is None:
                    t_dx_omega = eval_t_dx_omega(alpha, omega_F, dxbar_cache, Fx_cache)
            else:
                t_dx_omega = t_dx_omega_pred
                predicted_t_dx_omega[0] = False #for the next alpha, we have to
                # calculate it here again

            # check the two inequalities:
            if (rmt_eta_lower <= t_dx_omega and t_dx_omega <= rmt_eta_upper)\
                    or (rmt_eta_lower > t_dx_omega and alpha == 1.0):
                return 0.0
            elif rmt_eta_lower > t_dx_omega: #i.e. our step size is too small
                return -1.0
            elif t_dx_omega > rmt_eta_upper: #i.e. our step size is too big
                return 1.0


    # =================================================================
    ###################################################################
    # perform actual line search:
    ###################################################################
    # =================================================================



    ###################################################################
    # Define bracket for brentq rootfinding
    ###################################################################

    ###################################################################
    # possibly make a prediction:
    ###################################################################
    if 'old_omega' in parameters:
        old_omega = parameters['old_omega']
        alpha = min(
            1.0,
            rmt_eta / (old_omega * norm(dx))
        )
    else:
        # this is the case in the first iteration and if prediction == False
        alpha = 1.0

    # check whether the predicted step size is feasible, too big or too small
    # calculate t_dx_omega for the predicted step, which might be the full step:
    t_dx_omega_pred = eval_t_dx_omega(alpha, omega_F, dxbar_cache, Fx_cache)
    feas_flag = rmt_fct(alpha, omega_F, predicted_t_dx_omega, dxbar_cache, Fx_cache,
                        t_dx_omega=t_dx_omega_pred)

    predicted_t_dx_omega[0] = True # as we always start the bracket with the predicted alpha, we can reuse the values
    if feas_flag == 0:
        # prediction is feasible -> return
        if prediction:
            parameters['old_omega'] = omega_F[0]
        return alpha, dxbar_cache[0], Fx_cache[0]

    elif feas_flag == -1:
        # try to increase the predicted step size
        bracket = [alpha, 1.0]

    else: #feas_flag == 1:
        # try to reduce the predicted step size
        bracket = [alpha, 0.0]

    ###################################################################
    # call brentq procedure:
    ###################################################################
    result = root_scalar(rmt_fct, args=(omega_F, predicted_t_dx_omega, dxbar_cache, Fx_cache),
                         method='brentq', bracket=bracket, xtol=astall)

    ###################################################################
    # process result:
    ###################################################################
    if result.converged and result.root >= amin:
        if prediction:
            parameters['old_omega'] = omega_F[0]

        return result.root, dxbar_cache[0], Fx_cache[0]
    else:
        warn("\nThe restrictive monotonicity test could not determine a"
             " suitable step size!\n"
             "Take a full step instead, skip back projection (if set) and hope for the best.",
             LineSearchWarning)
        return None, dxbar_cache[0], Fx_cache[0]
### (IS)


#### (IS)
#------------------------------------------------------------------------------
# Backward step control based line search
#------------------------------------------------------------------------------
def scalar_search_bsc(func, x, dx, Fx, parameters = None):
    """
    Restrictive monotonicity test based line search as described in [1]_

    Parameters
    ----------
    func : callable
        Function returning J(x_k)^-1 F(x_k + alpha dx)
    x : ndarray
        Current iterate
    dx : ndarray
        Search direction
    Fx : ndarray:
        F(x_k)
    parameters : dict
        Can contain the following parameters:
        i)      rmt_eta : float
                    necessary tuning parameter (default 0.8)
        ii)     old_omega : float
                    Final curvature estimate of the previous iteration
        iii)    pred_init_frame : ndarray
                    If a prediction for the step size is made, this defines the brackets for the
                    root finding procedure, which then is [left_factor * prediction, right_factor * prediction]
                    (default: [0.9, 1.1])
        iv)     pred_frame_growth : float
                    Factor for the increase of the bracket size for the prediction of the step size, if the
                    previous bracket did not contain a solution (default: 1.2)
        v)      maxiter : int
                    Maximum number of calls to scipys scalar rootfinding procedure (default: 10)
        vi)     prediction : bool
                    Should the prediction for the step size be made? (default: True)
        vii)    amin : float
                    Minimum step size (default: 1e-6)

    Returns
    -------
    alpha : float
        Step length

    References
    ----------
    [1] Bock H.G., Kostina E., Schlöder J.P. (2000) "On the Role of Natural
        Level Functions to Achieve Global Convergence for Damped Newton
        Methods." In: Powell M.J.D., Scholtes S. (eds) System Modelling and
        Optimization. CSMO 1999. IFIP — The International Federation for
        Information Processing, vol 46. Springer, Boston, MA
    """

    #(LS)
    if parameters == None:
        print("Missing jacobian")
    prepare_parameters('bsc',parameters,parameters['jacobian'],dx)
    #(LS)

    # =================================================================
    ###################################################################
    # read and prepare parameters and variables:
    ###################################################################
    # =================================================================
    H = parameters['H']
    H_lower = parameters['H_lower']
    H_upper = parameters['H_upper']
    smooth = parameters['smooth_factor']
    amin = parameters['amin']
    astall = parameters['astall']

    # H_prime will store H_prime to make a prediction possible in the next iteration
    H_prime_container = [-1.0]
    Fx_cache = [None]
    #flag whether t_dx_omega was already computed for one of the boundaries of the bracket
    predicted_H_prime = [False]

    jacobian = parameters['jacobian']
    #unfortunately, we have to deepcopy the Jacobian as we have to reset the Jacobian after updating it for
    #intermediate step sizes and as some update schemes might not lead to the original Jacobian if we first
    #use .update(x_new) and then .update(x_old).
    old_jacobian = deepcopy(jacobian)
    jac_tol = parameters['jac_tol']


    # =================================================================
    ###################################################################
    # define auxiliary functions:
    ###################################################################
    # =================================================================

    ###################################################################
    # function to calculate H_prime:
    ###################################################################
    def eval_H_prime(alpha, H_prime_container, Fx_cache):
        x_new = x + alpha * dx
        Fx_new = func(x_new)
        jacobian = deepcopy(old_jacobian)
        jacobian.update(
            x_new.copy(),
            Fx_new
        )
        dx_next_it = -jacobian.solve(
            Fx_new,
            tol=jac_tol
        )
        parameters['dx_next'] = dx_next_it
        Fx_cache[0] = Fx_new
        dx_diff = dx_next_it - dx
        H_prime = alpha * norm(dx_diff)

        H_prime_container[0] = H_prime #store H_prime for next iteration

        return H_prime

    ###################################################################
    # function checking whether the BSC test is satisfied:
    ###################################################################
    # to make it callable by root_scalar, we construct it such that
    # i) for all step sizes satisfying
    #   H_lower <= H_prime <= H_upper
    #   we return 0.0, which aborts the root finding procedure and classifies
    #   the step size as feasible
    # ii) if the left inequality is violated, we return -1.0
    # iii) if the right inequality is violated, we return 1.0
    # This is necessary to be able to have different signs for the left and right end
    # of the interval in the bisection procedure.
    def bsc_fct(alpha, H_prime_container, predicted_H_prime, Fx_cache, H_prime=None):
        # although we call a root finding procedure, we only want to achieve
        # H_lower <= H_prime <= H_upper. Hence, we set a negative result to 0.0 to end the root
        # finding procedure.
        # to avoid that left and right interval boundary have the same sign, we
        # return -1 for too small values.
        if alpha < amin:
            if predicted_H_prime[0]:
                predicted_H_prime[0] = False
            return -1

        else:
            #if we already calculated t_dx_omega, reuse it
            if not predicted_H_prime[0]:
                if H_prime is None:
                    H_prime = eval_H_prime(alpha, H_prime_container, Fx_cache)
            else:
                H_prime = H_prime_container[0]
                predicted_H_prime[0] = False #for the next alpha, we have to
                # calculate it here again

            # check the two inequalities:
            if (H_lower <= H_prime and H_prime <= H_upper)\
                    or (H_lower > H_prime and alpha >= 1.0):
                return 0.0
            elif H_lower > H_prime and alpha < 1.0: #i.e. our step size is too small
                return -1.0
            elif H_prime > H_upper: #i.e. our step size is too big
                return 1.0


    # =================================================================
    ###################################################################
    # perform actual line search:
    ###################################################################
    # =================================================================

    ###################################################################
    # possibly make a prediction:
    ###################################################################
    if 'old_H_prime' in parameters and 'old_step_size' in parameters:
        old_H_prime = parameters['old_H_prime']
        old_step_size = parameters['old_step_size']
        alpha = min(
            1.0,
            old_step_size * (smooth + (1-smooth)*H/old_H_prime)
        )
    else:
        # this is the case in the first iteration and if prediction == False
        alpha = 1.0

    ###################################################################
    # check prediction and define bracket for brentq rootfinding:
    ###################################################################
    # check whether the step size is feasible, too big or too small
    # calculate t_dx_omega for the predicted step, which might be the full step:
    H_prime = eval_H_prime(alpha, H_prime_container, Fx_cache)
    feas_flag = bsc_fct(alpha, H_prime_container, predicted_H_prime, Fx_cache,
                        H_prime=H_prime)
    predicted_H_prime[0] = True

    if feas_flag == 0:
        # prediction is feasible
        parameters['old_H_prime'] = H_prime_container[0]
        parameters['old_step_size'] = alpha

        return alpha, Fx_cache[0]

    elif feas_flag == -1:
        # try to increase the predicted step size
        bracket = [alpha, 1.0]

    else: #feas_flag == 1:
        # try to reduce the predicted step size
        bracket = [alpha, 0.0]

    ###################################################################
    # call brentq procedure:
    ###################################################################
    result = root_scalar(bsc_fct, args=(H_prime_container, predicted_H_prime, Fx_cache),
                         method='brentq', bracket=bracket, xtol=astall)

    ###################################################################
    # process result:
    ###################################################################
    if result.converged and result.root >= amin:
        alpha = result.root
        parameters['old_H_prime'] = H_prime_container[0]
        parameters['old_step_size'] = alpha

        return alpha, Fx_cache[0]
    else:
        warn("\nThe backward step control could not determine a"
             " suitable step size!\n"
             "Take a full step instead and hope for the best.", LineSearchWarning)

        parameters['old_H_prime'] = H_prime_container[0]
        parameters['old_step_size'] = 1.0

        return None, Fx_cache[0]


### (IS)