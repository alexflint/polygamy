import os
from fractions import Fraction

import numpy as np

from spline import evaluate_zero_offset_bezier, evaluate_zero_offset_bezier_second_deriv
from utils import cayley, cayley_mat, cayley_denom, normalized, essential_matrix_from_relative_pose
from polynomial import Polynomial
from polynomial_io import write_polynomials, write_solution

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def chop(x, block_sizes):
    assert len(x) == sum(block_sizes)
    offset = 0
    for n in block_sizes:
        yield x[offset:offset+n]
        offset += n

def main():
    np.random.seed(1)

    #
    # Construct ground truth
    #
    num_frames = 5
    num_landmarks = 10
    num_imu_readings = 8
    bezier_degree = 3
    out = 'out/full_initialization'

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Bezier curve degree:', bezier_degree

    if not os.path.isdir(out):
        os.mkdir(out)

    # Both splines should start at 0,0,0
    frame_times = np.linspace(0, .9, num_frames)
    accel_times = np.linspace(0, 1, num_imu_readings)

    true_pos_controls = np.random.randn(bezier_degree-1, 3)
    true_orient_controls = np.random.randn(bezier_degree-1, 3)

    true_landmarks = np.random.randn(num_landmarks, 3)

    true_frame_positions = np.array([evaluate_zero_offset_bezier(true_pos_controls, t) for t in frame_times])
    true_frame_cayleys = np.array([evaluate_zero_offset_bezier(true_orient_controls, t) for t in frame_times])
    true_frame_orientations = np.array(map(cayley, true_frame_cayleys))

    true_imu_cayleys = np.array([evaluate_zero_offset_bezier(true_orient_controls, t) for t in accel_times])
    true_imu_orientations = np.array(map(cayley, true_imu_cayleys))

    true_gravity_magnitude = 9.8
    true_gravity = normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3)
    true_global_accels = np.array([evaluate_zero_offset_bezier_second_deriv(true_pos_controls, t) for t in accel_times])
    true_accels = np.array([np.dot(R, a + true_gravity) + true_accel_bias
                            for R, a in zip(true_imu_orientations, true_global_accels)])

    true_features = np.array([[normalized(np.dot(R, x-p)) for x in true_landmarks]
                              for R, p in zip(true_frame_orientations, true_frame_positions)])

    true_vars = np.hstack((true_pos_controls.flatten(),
                           true_orient_controls.flatten(),
                           true_accel_bias,
                           true_gravity))

    print np.min(true_features.reshape((-1, 3)), axis=0)
    print np.max(true_features.reshape((-1, 3)), axis=0)

    #
    # Add sensor noise
    #

    accel_noise = 0
    feature_noise = 0

    observed_features = true_features.copy()
    observed_accels = true_accels.copy()

    if accel_noise > 0:
        observed_accels += np.random.randn(*observed_accels.shape) * accel_noise

    if feature_noise > 0:
        observed_features += np.random.rand(*observed_features.shape) * feature_noise

    #
    # Construct symbolic versions of the above
    #
    num_position_vars = (bezier_degree-1)*3
    num_orientation_vars = (bezier_degree-1)*3
    num_accel_bias_vars = 3
    num_gravity_vars = 3

    block_sizes = [num_position_vars, num_orientation_vars, num_accel_bias_vars, num_gravity_vars]
    num_vars = sum(block_sizes)

    sym_vars = [Polynomial.coordinate(i, num_vars, Fraction) for i in range(num_vars)]
    sym_pos_controls, sym_orient_controls, sym_accel_bias, sym_gravity = map(np.array, chop(sym_vars, block_sizes))

    sym_pos_controls = sym_pos_controls.reshape((-1, 3))
    sym_orient_controls = sym_orient_controls.reshape((-1, 3))

    assert len(true_vars) == len(sym_vars)

    #
    # Accel residuals
    #
    residuals = []

    print 'Accel residuals:'
    for i, t in enumerate(accel_times):
        sym_cayley = evaluate_zero_offset_bezier(sym_orient_controls, t)
        sym_orient = cayley_mat(sym_cayley)
        sym_denom = cayley_denom(sym_cayley)
        sym_global_accel = evaluate_zero_offset_bezier_second_deriv(sym_pos_controls, t)
        sym_accel = np.dot(sym_orient, sym_global_accel + sym_gravity) + sym_denom * sym_accel_bias
        residual = sym_accel - sym_denom * observed_accels[i]
        residuals.extend(residual)
        for r in residual:
            print '  %f   (degree=%d, length=%d)' % (r(*true_vars), r.total_degree, len(r))

    #
    # Epipolar residuals
    #

    print 'Epipolar residuals:'
    for i, ti in enumerate(frame_times):
        if i == 0: continue
        sym_Ri = cayley_mat(evaluate_zero_offset_bezier(sym_orient_controls, ti))
        sym_pi = evaluate_zero_offset_bezier(sym_pos_controls, ti)
        sym_E = essential_matrix_from_relative_pose(sym_Ri, sym_pi)
        for k in range(num_landmarks):
            z1 = observed_features[0][k]
            zi = observed_features[i][k]
            residual = np.dot(zi, np.dot(sym_E, z1))
            residuals.append(residual)
            r = residual
            print '  %f   (degree=%d, length=%d)' % (r(*true_vars), r.total_degree, len(r))

    #
    # Construct cost
    #

    cost = Polynomial(num_vars)
    for r in residuals:
        cost += r*r

    gradients = cost.partial_derivatives()

    print '\nNum vars:', num_vars
    print 'Num residuals:', len(residuals)
    print '\nCost:'
    print '  Num terms: %d' % len(cost)
    print '  Degree: %d' % cost.total_degree


    #
    # Output to file
    #
    write_polynomials(cost, out+'/cost.txt')
    write_polynomials(residuals, out+'/residuals.txt')
    write_polynomials(gradients, out+'/gradients.txt')
    write_solution(true_vars, out+'/solution.txt')

    np.savetxt(out+'/feature_measurements.txt', observed_features.reshape((-1, 3)))
    np.savetxt(out+'/accel_measurements.txt', observed_accels)
    np.savetxt(out+'/problem_size.txt', [num_frames, num_landmarks, num_imu_readings])
    np.savetxt(out+'/frame_times.txt', frame_times)
    np.savetxt(out+'/accel_times.txt', accel_times)

    np.savetxt(out+'/true_pos_controls.txt', true_pos_controls)
    np.savetxt(out+'/true_orient_controls.txt', true_orient_controls)
    np.savetxt(out+'/true_accel_bias.txt', true_accel_bias)
    np.savetxt(out+'/true_gravity.txt', true_gravity)


    return

    #
    # Plot
    #
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ts = np.linspace(0, 1, 100)
    true_ps = np.array([evaluate_zero_offset_bezier(true_pos_controls, t) for t in ts])

    ax.plot(true_ps[:, 0], true_ps[:, 1], true_ps[:, 2], '-b')

    plt.show()


if __name__ == '__main__':
    main()
