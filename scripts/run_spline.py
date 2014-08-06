__author__ = 'alexflint'

import StringIO
from fractions import Fraction
import numpy as np

from polynomial import *
from spline import *
import action
import multivariate


def skew(m):
    m = np.asarray(m)
    return np.array([[  0,    -m[2],  m[1] ],
                     [  m[2],  0,    -m[0] ],
                     [ -m[1],  m[0],  0.   ]])


def is_squarefree(fs, zeros):
    J = polynomial_jacobian(fs)
    for zero in zeros:
        Jcur = J(*zero)
        print Jcur


def array_str(arr):
    strings = []
    maxlen = 0
    for row in arr:
        rowstrings = []
        for elem in row:
            if isinstance(elem, Polynomial):
                s = elem.format(compact=True)
            else:
                s = unicode(elem)
            rowstrings.append(s)
            maxlen = max(maxlen, len(s))
        strings.append(rowstrings)

    ss = StringIO.StringIO()
    ss.write('[')
    for i,rowstrings in enumerate(strings):
        if i > 0:
            ss.write('\n ')
        ss.write('[')
        for j,s in enumerate(rowstrings):
            if j > 0:
                ss.write('  ')
            ss.write(s)
            ss.write(' '*(maxlen-len(s)))
        ss.write(']')
    ss.write(']')
    return ss.getvalue()


def arraymap(f, x):
    if isinstance(x, Polynomial):
         return f(x)
    else:
        try:
            return np.array(map(lambda xi: arraymap(f, xi), x))
        except:
            return f(x)


def astype(x, ctype):
    return arraymap(lambda xi: xi.astype(ctype) if isinstance(xi, Polynomial) else ctype(xi), x)


def asfraction(x, denom_limit=None):
    return arraymap(lambda xi: Fraction(xi).limit_denominator(denom_limit), x)


def flatten(x):
    L = []
    for xi in x:
        if isinstance(xi, Polynomial):
            L.append(xi)
        else:
            try:
                L.extend(xi)
            except:
                L.append(xi)
    return L


def cayley_mat(s):
    return np.eye(3) * (1 - np.dot(s,s)) + 2*skew(s) + 2*np.outer(s,s)


def cayley_denom(s):
    return 1 + np.dot(s,s)


def cayley(s):
    return cayley_mat(s) / cayley_denom(s)


def main():
    np.random.seed(123)
    np.set_printoptions(precision=5, suppress=True, linewidth=300)
    ordering = GrevlexOrdering()

    # Construct symbolic problem
    num_landmarks = 4
    nv = 1 + 4 + 4 + num_landmarks * 3
    params = [ Polynomial.coordinate(i, nv, Fraction) for i in range(nv) ]
    time = params[0]
    p_controls = params[1:5]
    s_controls = params[5:9]
    landmarks = [params[9+i*3:9+i*3+3] for i in range(0, num_landmarks)]

    p = bezier(p_controls, time)
    v = p.partial_derivative(0)
    a = v.partial_derivative(0)

    s = bezier(s_controls, time)
    
    # Sample ground truth
    true_times = [ Fraction(1,8), Fraction(4,8), Fraction(6,8), Fraction(7,8), Fraction(5,8) ]
    true_params = np.random.rand(nv)

    # Construct the least squares spline cost
    cost = Polynomial(nv, Fraction)
    for ti in zip(true_times, noisy_ys):
        pi = p.evaluate_partial(0, ti).drop(0)
        si = s.evaluate_partial(0, ti).dtop(0)
        Ri = cayley_mat(s)  # TODO: denominator!!

        for xi in landmarks:
            yi = np.dot(Ri, xi-pi)

        true_pi = [ p(t, *true_params) for t in true_times ]

        

        cost += residual**2



    # Construct the least squares cost for rotation estimation
    nv = 3
    coords = np.array([ Polynomial.coordinate(i, nv, Fraction) for i in range(nv)])
    s = coords[:3]
    t = coords[3:]

    Q = astype(cayley_mat(s), Fraction)
    k = astype(cayley_denom(s), Fraction)
    #h1 = t[0] * (1 + np.dot(s,s)) - 1  # auxiliary equation needed to remove (1-s*s)=0
    #h2 = t[1] * (1 - np.dot(s,s)) - 1  # auxiliary equation needed to remove (1-s*s)=0

    true_s = [Fraction(1,2), Fraction(2,3), Fraction(-3,4)]
    #true_s = [Fraction(0), Fraction(0), Fraction(0)]

    true_t = [] #Fraction(1, 1+np.dot(true_s,true_s))] #, Fraction(1, 1-np.dot(true_s,true_s))]

    true_Q = cayley_mat(true_s)
    true_R = cayley(true_s)

    true_us = astype(np.random.randint(-10, 10, size=(7,3)), Fraction)
    #true_us = astype(np.eye(3), Fraction)

    true_vs = asfraction([np.dot(true_R, u) for u in true_us], 1000)

    true_params = true_s + true_t

    print '\n'
    print 'Q:'
    print array_str(Q)
    print 'k:',k
    print 'us:'
    print true_us.astype(float)
    print 'vs:'
    print true_vs.astype(float)
    print 'true_R:'
    print true_R.astype(float)
    print 'true_Q:'
    print true_Q.astype(float)

    residuals = flatten(np.dot(Q,u)-k*v for u,v in zip(true_us,true_vs))

    auxiliary_system = [] #h1] #h1, h2, t[0]-true_t[0]] #s[0]-true_s[0], s[1]-true_s[1], s[2]-true_s[2]]
    auxiliary_system = flatten(Q - astype(cayley_mat(true_s), Fraction))

    print '\nresiduals:'
    for ri in residuals:
        print '  ',ri

    cayley_cost = sum(r**2 for r in residuals)

    print 'cayley_cost:',cayley_cost

    cost = cayley_cost
    #optimization_system = [cost.partial_derivative(i) for i in range(3)] + auxiliary_system
    optimization_system = auxiliary_system
    print '\nOptimization cost:', cost
    print '\nOptimization system:'
    for f in optimization_system:
        print '  ', f



    F = optimization_system

    # Evaluate at (hopefully unique) zero
    fun = polynomial_vector(F)
    Jfun = polynomial_jacobian(F)
    J = Jfun(*true_params)
    print '\nEvaluated at ground truth:'
    print fun(*true_params)
    print 'Jacobian at ground truth:'
    print J
    print 'Jacobian singular values:'
    singular_values = np.linalg.svd(J, compute_uv=False)
    print singular_values

    # Compute grobner basis
    print '\nComputing grobner basis...'
    G = gbasis(F, ordering, limit=100)

    #print 'Checking grobner basis...'
    #assert is_grobner_basis(G, ordering)

    print 'Grobner basis:'
    for g in G:
        print '  ',g

    return

    # Compute action matrix
    soln = []
    for i in range(0,nv):
        coord = Polynomial.coordinate(i, nv, Fraction)
        M = action.action_matrix_from_grobner_basis(coord, G, ordering)

        eigvals, eigvecs = np.linalg.eig(M.astype(float))
        print 'Coordinate %d:' % i
        print eigvals

        if len(eigvals) == 1:
            soln.append(eigvals[0])

    #report = multivariate.polish_root(optimization_system, soln)
    #print '\nPolished solution:'
    #print report.x


if __name__ == '__main__':
    main()
