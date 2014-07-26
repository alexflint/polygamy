import os
from fractions import Fraction

import numpy as np
import scipy.optimize

from polynomial import Polynomial, parse
from spline import evaluate_zero_offset_bezier, evaluate_zero_offset_bezier_second_deriv
from utils import cayley, cayley_mat, cayley_denom, skew
import compilation

def normalized(x):
    x = np.asarray(x)
    return x / np.linalg.norm(x)


def essential_matrix(R1, p1, R2, p2):
    Rrel = np.dot(R2, R1.T)
    prel = np.dot(R1, p2-p1)
    return np.dot(Rrel, skew(prel))


def epipolar():
    np.random.rand(434)

    s1, s2 = np.random.rand(2, 3)
    R1, R2 = map(cayley, (s1, s2))
    p1, p2 = np.random.rand(2, 3)

    E = essential_matrix(R1, p1, R2, p2)

    x = np.random.rand(3)
    y1 = np.dot(R1, x-p1)
    y2 = np.dot(R2, x-p2)

    print np.dot(y2, np.dot(E, y1))


def run_sfm():
    # Construct symbolic problem
    num_landmarks = 4
    num_frames = 2

    print 'Num observations: ', num_landmarks * num_frames * 2
    print 'Num vars: ', num_frames*6 + num_landmarks*3 + num_frames*num_landmarks

    true_landmarks = np.random.randn(num_landmarks, 3)
    true_positions = np.random.rand(num_frames, 3)
    true_cayleys = np.random.rand(num_frames, 3)

    true_qs = map(cayley_mat, true_cayleys)
    true_betas = map(cayley_denom, true_cayleys)
    true_rotations = [(q/b) for (q,b) in zip(true_qs, true_betas)]

    true_uprojections = [[np.dot(R, x-p) for x in true_landmarks]
                         for R,p in zip(true_rotations, true_positions)]

    true_projections = [[normalized(zu) for zu in row] for row in true_uprojections]
    true_alphas = [[np.linalg.norm(zu) for zu in row] for row in true_uprojections]

    true_vars = np.hstack((true_cayleys.flatten(),
                           true_positions.flatten(),
                           true_landmarks.flatten(),
                           np.asarray(true_alphas).flatten()))

    #true_projection_mat = np.reshape(true_projections, (num_frames, num_landmarks, 2))

    for i in range(num_frames):
        p = true_positions[i]
        q = true_qs[i]
        beta = true_betas[i]
        for j in range(num_landmarks):
            x = true_landmarks[j]
            z = true_projections[i][j]
            alpha = true_alphas[i][j]
            print alpha * beta * z - np.dot(q, x-p)

    # construct symbolic versions of the above
    s_offs = 0
    p_offs = s_offs + num_frames*3
    x_offs = p_offs + num_frames*3
    a_offs = x_offs + num_landmarks*3
    num_vars = a_offs + num_landmarks*num_frames

    vars = [Polynomial.coordinate(i, num_vars, Fraction) for i in range(num_vars)]
    sym_cayleys = np.reshape(vars[s_offs:s_offs+num_frames*3], (num_frames, 3))
    sym_positions = np.reshape(vars[p_offs:p_offs+num_frames*3], (num_frames, 3))
    sym_landmarks = np.reshape(vars[x_offs:x_offs+num_landmarks*3], (num_landmarks, 3))
    sym_alphas = np.reshape(vars[a_offs:], (num_frames, num_landmarks))

    residuals = []
    for i in range(num_frames):
        sym_p = sym_positions[i]
        sym_s = sym_cayleys[i]
        for j in range(num_landmarks):
            sym_x = sym_landmarks[j]
            sym_a = sym_alphas[i,j]
            true_z = true_projections[i][j]
            residual = np.dot(cayley_mat(sym_s), sym_x-sym_p) - sym_a * cayley_denom(sym_s) * true_z
            residuals.extend(residual)

    print 'Residuals:'
    cost = Polynomial(num_vars)
    for residual in residuals:
        cost += np.dot(residual, residual)
        print '  ',residual(*true_vars)  #ri.num_vars, len(true_vars)

    print '\nGradients:'
    gradient = [cost.partial_derivative(i) for i in range(num_vars)]
    for gi in gradient:
        print gi(*true_vars)

    J = np.array([[r.partial_derivative(i)(*true_vars) for i in range(num_vars)]
                  for r in residuals])

    print '\nJacobian singular values:'
    print J.shape
    U,S,V = np.linalg.svd(J)
    print S

    print '\nHessian eigenvalues:'
    H = np.dot(J.T, J)
    print H.shape
    print np.linalg.eigvals(H)


def run_epipolar():
    # Construct symbolic problem
    num_landmarks = 10
    num_frames = 3

    true_landmarks = np.random.randn(num_landmarks, 3)
    true_positions = np.vstack((np.zeros(3),
                                np.random.rand(num_frames-1, 3)))
    true_cayleys = np.vstack((np.zeros(3),
                              np.random.rand(num_frames-1, 3)))

    true_qs = map(cayley_mat, true_cayleys)
    true_rotations = map(cayley, true_cayleys)

    true_uprojections = [[np.dot(R, x-p) for x in true_landmarks]
                         for R,p in zip(true_rotations, true_positions)]

    true_projections = [[normalized(zu) for zu in row] for row in true_uprojections]

    p0 = true_positions[0]
    q0 = true_qs[0]
    for i in range(1, num_frames):
        p = true_positions[i]
        q = true_qs[i]
        E = essential_matrix(q0, p0, q, p)
        for j in range(num_landmarks):
            z = true_projections[i][j]
            z0 = true_projections[0][j]
            print np.dot(z, np.dot(E, z0))

    # construct symbolic versions of the above
    s_offs = 0
    p_offs = s_offs + (num_frames-1)*3
    num_vars = p_offs + (num_frames-1)*3

    vars = [Polynomial.coordinate(i, num_vars, Fraction) for i in range(num_vars)]
    sym_cayleys = np.reshape(vars[s_offs:s_offs+(num_frames-1)*3], (num_frames-1, 3))
    sym_positions = np.reshape(vars[p_offs:p_offs+(num_frames-1)*3], (num_frames-1, 3))

    true_vars = np.hstack((true_cayleys[1:].flatten(),
                           true_positions[1:].flatten()))

    residuals = []
    p0 = np.zeros(3)
    R0 = np.eye(3)
    for i in range(1, num_frames):
        sym_p = sym_positions[i-1]
        sym_s = sym_cayleys[i-1]
        sym_q = cayley_mat(sym_s)
        sym_E = essential_matrix(R0, p0, sym_q, sym_p)
        for j in range(num_landmarks):
            z = true_projections[i][j]
            z0 = true_projections[0][j]
            residual = np.dot(z, np.dot(sym_E, z0))
            print 'Residual poly: ',len(residual), residual.total_degree
            residuals.append(residual)

    print 'Num vars:',num_vars
    print 'Num residuals:',len(residuals)

    print 'Residuals:', len(residuals)
    cost = Polynomial(num_vars)
    for residual in residuals:
        #cost += np.dot(residual, residual)
        print '  ',residual(*true_vars)  #ri.num_vars, len(true_vars)

    print '\nGradients:'
    gradient = [cost.partial_derivative(i) for i in range(num_vars)]
    for gi in gradient:
        print '  ',gi(*true_vars)

    J = np.array([[r.partial_derivative(i)(*true_vars) for i in range(num_vars)]
                  for r in residuals])

    print '\nJacobian singular values:'
    print J.shape
    U,S,V = np.linalg.svd(J)
    print S
    print V[-1]
    print V[-2]

    print '\nHessian eigenvalues:'
    H = np.dot(J.T, J)
    print H.shape
    print np.linalg.eigvals(H)


def run_spline_epipolar():
    # Construct symbolic problem
    num_landmarks = 10
    num_frames = 3
    num_imu_readings = 8
    bezier_degree = 4
    out = 'out/epipolar_accel_bezier3'

    if not os.path.isdir(out):
        os.mkdir(out)

    # Both splines should start at 0,0,0
    frame_times = np.linspace(0, .9, num_frames)
    imu_times = np.linspace(0, 1, num_imu_readings)

    true_rot_controls = np.random.rand(bezier_degree-1, 3)
    true_pos_controls = np.random.rand(bezier_degree-1, 3)

    true_landmarks = np.random.randn(num_landmarks, 3)
    true_cayleys = np.array([evaluate_zero_offset_bezier(true_rot_controls, t) for t in frame_times])
    true_positions = np.array([evaluate_zero_offset_bezier(true_pos_controls, t) for t in frame_times])

    true_accels = np.array([evaluate_zero_offset_bezier_second_deriv(true_pos_controls, t) for t in imu_times])

    true_qs = map(cayley_mat, true_cayleys)
    true_rotations = map(cayley, true_cayleys)

    true_uprojections = [[np.dot(R, x-p) for x in true_landmarks]
                         for R,p in zip(true_rotations, true_positions)]

    true_projections = [[normalized(zu) for zu in row] for row in true_uprojections]

    p0 = true_positions[0]
    q0 = true_qs[0]
    for i in range(1, num_frames):
        p = true_positions[i]
        q = true_qs[i]
        E = essential_matrix(q0, p0, q, p)
        for j in range(num_landmarks):
            z = true_projections[i][j]
            z0 = true_projections[0][j]
            #print np.dot(z, np.dot(E, z0))

    # construct symbolic versions of the above
    s_offs = 0
    p_offs = s_offs + (bezier_degree-1)*3
    num_vars = p_offs + (bezier_degree-1)*3

    vars = [Polynomial.coordinate(i, num_vars, Fraction) for i in range(num_vars)]
    sym_rot_controls = np.reshape(vars[s_offs:s_offs+(bezier_degree-1)*3], (bezier_degree-1, 3))
    sym_pos_controls = np.reshape(vars[p_offs:p_offs+(bezier_degree-1)*3], (bezier_degree-1, 3))

    true_vars = np.hstack((true_rot_controls.flatten(),
                           true_pos_controls.flatten()))

    residuals = []

    # Accel residuals
    for i in range(num_imu_readings):
        sym_a = evaluate_zero_offset_bezier_second_deriv(sym_pos_controls, imu_times[i])
        residual = sym_a - true_accels[i]
        residuals.extend(residual)

    # Epipolar residuals
    p0 = np.zeros(3)
    R0 = np.eye(3)
    for i in range(1, num_frames):
        sym_s = evaluate_zero_offset_bezier(sym_rot_controls, frame_times[i])
        sym_p = evaluate_zero_offset_bezier(sym_pos_controls, frame_times[i])
        sym_q = cayley_mat(sym_s)
        #sym_q = np.eye(3) * (1. - np.dot(sym_s, sym_s)) + 2.*skew(sym_s) + 2.*np.outer(sym_s, sym_s)
        sym_E = essential_matrix(R0, p0, sym_q, sym_p)
        for j in range(num_landmarks):
            z = true_projections[i][j]
            z0 = true_projections[0][j]
            residual = np.dot(z, np.dot(sym_E, z0))
            residuals.append(residual)

    print 'Num vars:',num_vars
    print 'Num residuals:',len(residuals)

    print 'Residuals:', len(residuals)
    cost = Polynomial(num_vars)
    for r in residuals:
        cost += r*r
        print '  %f   (degree=%d, length=%d)' % (r(*true_vars), r.total_degree, len(r))

    print '\nCost:'
    print '  Num terms: %d' % len(cost)
    print '  Degree: %d' % cost.total_degree

    print '\nGradients:'
    gradients = cost.partial_derivatives()
    for gradient in gradients:
        print '  %d  (degree=%d, length=%d)' % (gradient(*true_vars), gradient.total_degree, len(gradient))

    jacobians = [r.partial_derivatives() for r in residuals]

    J = np.array([[J_ij(*true_vars) for J_ij in row] for row in jacobians])

    U, S, V = np.linalg.svd(J)

    print '\nJacobian singular values:'
    print J.shape
    print S
    null_space_dims = sum(np.abs(S) < 1e-5)
    if null_space_dims > 0:
        print '\nNull space:'
        for i in null_space_dims:
            print V[-i]
            print V[-2]

    print '\nHessian eigenvalues:'
    H = np.dot(J.T, J)
    print H.shape
    print np.linalg.eigvals(H)

    # Output to file
    with open(out+'/gradients.txt', 'w') as fd:
        for gradient in gradients:
            fd.write(gradient.format(use_superscripts=False) + ';\n')

    with open(out+'/residuals.txt', 'w') as fd:
        for residual in residuals:
            fd.write(residual.format(use_superscripts=False) + ';\n')

    with open(out+'/jacobians.txt', 'w') as fd:
        for row in jacobians:
            for j in row:
                fd.write(j.format(use_superscripts=False) + ';\n')

    with open(out+'/cost.txt', 'w') as fd:
        fd.write(cost.format(use_superscripts=False) + ';\n')

    with open(out+'/solution.txt', 'w') as fd:
        for i, xi in enumerate(true_vars):
            fd.write('x%d %.12f\n' % (i, xi))


def load_polynomials(path):
    return parse(*[line.strip('; \n').replace('^', '**') for line in open(path)])


def load_functions(path, varnames):
    return [compilation.function_from_expression(line.strip('; \n').replace('^', '**'), varnames)
            for line in open(path)]


def load_solution(path):
    varnames = []
    solution = []
    with open(path) as fd:
        for line in fd:
            var, value = line.strip().split()
            varnames.append(var)
            solution.append(float(value))
    return varnames, np.array(solution)


def analyze_polynomial():
    print 'Loading polynomials...'
    cost = load_polynomials('out/epipolar_accel_Bezier3_cost.txt')
    print '  Done loading.'

    assert isinstance(cost, Polynomial)
    print len(cost), cost.total_degree


def evaluate_array(arr, x):
    arr = np.asarray(arr)
    return np.array([a(*x) for a in arr.flat]).reshape(arr.shape)


def analyze_polynomial2():
    print 'Loading polynomials...'
    varnames, true_values = load_solution('out/epipolar_accel_bezier3/solution.txt')
    cost = load_functions('out/epipolar_accel_bezier3/cost.txt', varnames)[0]
    residuals = load_functions('out/epipolar_accel_bezier3/residuals.txt', varnames)
    jacobians = load_functions('out/epipolar_accel_bezier3/jacobians.txt', varnames)
    gradients = load_functions('out/epipolar_accel_bezier3/gradients.txt', varnames)

    residuals = np.array(residuals)
    jacobians = np.array(jacobians).reshape((-1, len(varnames)))

    def J(x):
        print ' ... jacobian'
        return evaluate_array(jacobians, x)

    def r(x):
        print ' ... residual'
        return evaluate_array(residuals, x)

    np.random.seed(765)
    seed_values = true_values + np.random.rand(len(true_values)) * 100

    out = scipy.optimize.leastsq(func=r, x0=seed_values, Dfun=J, full_output=True)
    opt_values = out[0]

    print '\nGradients:'
    print evaluate_array(gradients, opt_values)
    print evaluate_array(gradients, true_values)

    print '\nCosts:'
    print cost(*opt_values)
    print cost(*true_values)

    print '\nJacobian singular values:'
    jac = J(opt_values)
    U, S, V = np.linalg.svd(jac)
    print S

    print '\nTrue:'
    print true_values

    print '\nSeed:'
    print seed_values

    print '\nOptimized:'
    print opt_values

    print '\nError:'
    print np.linalg.norm(opt_values - true_values)


def main():
    np.random.seed(123)
    np.set_printoptions(precision=5, suppress=True, linewidth=300)

    #analyze_polynomial()
    analyze_polynomial2()

    #run_spline_epipolar()
    #run_epipolar()
    #run_sfm()

def profile_main():
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    try:
        main()
    finally:
        pr.disable()
        pstats.Stats(pr).sort_stats('tottime').print_stats(30)


if __name__ == '__main__':
    main()
