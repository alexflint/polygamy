import os
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from spline import evaluate_zero_offset_bezier, evaluate_zero_offset_bezier_second_deriv, evaluate_bezier, evaluate_bezier_second_deriv
from utils import cayley, normalized, essential_matrix
from polynomial import Polynomial, quadratic_form
from lie import SO3


class BezierCurve(object):
    def __init__(self, controls):
        self._controls = controls

    def evaluate(self, t):
        return evaluate_bezier(self._controls, t)

    def second_derivative(self, t):
        return evaluate_bezier_second_deriv(self._controls, t)


class SensorModel(object):
    # feature sigma
    # accel sigma
    # gyro sigma
    # ignore drift for now
    pass


class VisualInertialMeasurements(object):
    # frame timestamps
    # features
    # imu timestamps
    # accel
    # gyro
    pass


class StructureAndMotionEstimate(object):
    # landmarks
    # bezier object
    # accel bias
    # gyro bias
    # gravity
    # from vector (needs to know bezier degree)
    # to vector
    pass


class StructureAndMotionProblem(object):
    # init with measurements and bezier degree
    # construct estimate from vector
    # evaluate cost
    # evaluate cost given vector
    # evaluate residuals
    # evaluate residuals given vector
    # evaluate jacobian
    # evaluate jacobian given vector
    pass


class PositionEstimate(object):
    # bezier object
    # accel bias
    # gravity
    # from vector (needs to know bezier degree)
    # to vector
    pass


class EpipolarPositionProblem(object):
    # init with measurements and bezier degree
    # construct estimate from vector
    # evaluate cost
    # evaluate cost given vector
    # evaluate residuals
    # evaluate residuals given vector
    # evaluate jacobian
    # evaluate jacobian given vector
    pass


def main():
    np.random.seed(1)

    #
    # Construct ground truth
    #
    num_frames = 5
    num_landmarks = 50
    num_imu_readings = 80
    bezier_degree = 4
    out = 'out/position_only_bezier3'

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Bezier curve degree:', bezier_degree

    if not os.path.isdir(out):
        os.mkdir(out)

    # Both splines should start at 0,0,0
    frame_times = np.linspace(0, .9, num_frames)
    imu_times = np.linspace(0, 1, num_imu_readings)

    true_rot_controls = np.random.randn(bezier_degree-1, 3)
    true_pos_controls = np.random.randn(bezier_degree-1, 3)

    true_landmarks = np.random.randn(num_landmarks, 3)

    true_frame_cayleys = np.array([evaluate_zero_offset_bezier(true_rot_controls, t) for t in frame_times])
    true_frame_orientations = np.array(map(cayley, true_frame_cayleys))
    true_frame_positions = np.array([evaluate_zero_offset_bezier(true_pos_controls, t) for t in frame_times])

    true_imu_cayleys = np.array([evaluate_zero_offset_bezier(true_rot_controls, t) for t in imu_times])
    true_imu_orientations = np.array(map(cayley, true_imu_cayleys))

    true_gravity_magnitude = 9.8
    true_gravity = normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3)
    true_global_accels = np.array([evaluate_zero_offset_bezier_second_deriv(true_pos_controls, t) for t in imu_times])
    true_accels = np.array([np.dot(R, a + true_gravity) + true_accel_bias
                            for R, a in zip(true_imu_orientations, true_global_accels)])

    true_features = np.array([[normalized(np.dot(R, x-p)) for x in true_landmarks]
                              for R, p in zip(true_frame_orientations, true_frame_positions)])

    print np.min(true_features.reshape((-1, 3)), axis=0)
    print np.max(true_features.reshape((-1, 3)), axis=0)

    #
    # Add sensor noise
    #

    accel_noise = 0#0.001
    feature_noise = 0#0.01
    orientation_noise = 0.01

    observed_frame_orientations = true_frame_orientations.copy()
    observed_imu_orientations = true_imu_orientations.copy()
    observed_features = true_features.copy()
    observed_accels = true_accels.copy()

    if orientation_noise > 0:
        for i, R in enumerate(observed_frame_orientations):
            R_noise = SO3.exp(np.random.randn(3)*orientation_noise)
            observed_frame_orientations[i] = np.dot(R_noise, R)
        for i, R in enumerate(observed_imu_orientations):
            R_noise = SO3.exp(np.random.randn(3)*orientation_noise)
            observed_imu_orientations[i] = np.dot(R_noise, R)

    if accel_noise > 0:
        observed_accels += np.random.randn(*observed_accels.shape) * accel_noise

    if feature_noise > 0:
        observed_features += np.random.rand(*observed_features.shape) * feature_noise

    #
    # Construct symbolic versions of the above
    #
    position_offs = 0
    accel_bias_offset = position_offs + (bezier_degree-1)*3
    gravity_offset = accel_bias_offset + 3
    num_vars = gravity_offset + 3

    sym_vars = [Polynomial.coordinate(i, num_vars, Fraction) for i in range(num_vars)]
    sym_pos_controls = np.reshape(sym_vars[position_offs:position_offs+(bezier_degree-1)*3], (bezier_degree-1, 3))
    sym_accel_bias = np.asarray(sym_vars[accel_bias_offset:accel_bias_offset+3])
    sym_gravity = np.asarray(sym_vars[gravity_offset:gravity_offset+3])

    true_vars = np.hstack((true_pos_controls.flatten(), true_accel_bias, true_gravity))
    assert len(true_vars) == len(sym_vars)

    #
    # Accel residuals
    #
    residuals = []

    print '\nAccel residuals:'
    for i in range(num_imu_readings):
        observed_R = observed_imu_orientations[i]
        sym_global_accel = evaluate_zero_offset_bezier_second_deriv(sym_pos_controls, imu_times[i])
        sym_accel = np.dot(observed_R, sym_global_accel + sym_gravity) + sym_accel_bias
        residual = sym_accel - observed_accels[i]
        residuals.extend(residual)

    #
    # Epipolar residuals
    #

    for i in range(num_frames):
        observed_Ri = observed_frame_orientations[i]
        sym_pi = evaluate_zero_offset_bezier(sym_pos_controls, frame_times[i])
        for j in range(num_frames):
            observed_Rj = observed_frame_orientations[j]
            sym_pj = evaluate_zero_offset_bezier(sym_pos_controls, frame_times[j])
            if i == j: continue
            sym_E = essential_matrix(observed_Ri, sym_pi, observed_Rj, sym_pj)
            for k in range(num_landmarks):
                zi = observed_features[i][k]
                zj = observed_features[j][k]
                residual = np.dot(zj, np.dot(sym_E, zi))
                residuals.append(residual)

    print '\nNum vars:', num_vars
    print 'Num residuals:', len(residuals)

    print '\nResiduals:', len(residuals)
    cost = Polynomial(num_vars)
    for r in residuals:
        cost += r*r
        print '  %f   (degree=%d, length=%d)' % (r(*true_vars), r.total_degree, len(r))

    print '\nCost:'
    print '  Num terms: %d' % len(cost)
    print '  Degree: %d' % cost.total_degree

    # Solve
    A, b, k = quadratic_form(cost)
    estimated_vars = np.squeeze(np.linalg.solve(A*2, -b))
    estimated_pos_controls = np.reshape(estimated_vars[position_offs:position_offs+(bezier_degree-1)*3], (bezier_degree-1, 3))
    estimated_positions = np.array([evaluate_zero_offset_bezier(estimated_pos_controls, t) for t in frame_times])
    estimated_accel_bias = np.asarray(estimated_vars[accel_bias_offset:accel_bias_offset+3])
    estimated_gravity = np.asarray(estimated_vars[gravity_offset:gravity_offset+3])
    re_estimated_gravity = normalized(estimated_gravity) * true_gravity_magnitude;

    print '\nEstimated:'
    print estimated_vars

    print '\nGround truth:'
    print true_vars

    print '\nTotal Error:', np.linalg.norm(estimated_vars - true_vars)
    print 'Accel bias error:', np.linalg.norm(estimated_accel_bias - true_accel_bias)
    print 'Gravity error:', np.linalg.norm(estimated_gravity - true_gravity)
    print '  True gravity:', true_gravity
    print '  Estimated gravity:', estimated_gravity
    print '  Estimated gravity magnitude:', np.linalg.norm(estimated_gravity)
    print '  Re-normalized gravity error: ', np.linalg.norm(re_estimated_gravity - true_gravity)
    for i in range(num_frames):
        print 'Frame %d error: %f' % (i, np.linalg.norm(estimated_positions[i] - true_frame_positions[i]))

    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ts = np.linspace(0, 1, 100)
    true_ps = np.array([evaluate_zero_offset_bezier(true_pos_controls, t) for t in ts])
    estimated_ps = np.array([evaluate_zero_offset_bezier(estimated_pos_controls, t) for t in ts])

    ax.plot(true_ps[:, 0], true_ps[:, 1], true_ps[:, 2], '-b')
    ax.plot(estimated_ps[:, 0], estimated_ps[:, 1], estimated_ps[:, 2], '-r')

    plt.show()

    # TODO:
    # - get real dataset
    # - python wrappers for feature tracker
    # - python wrappers for five point algorithm
    # - build up problem matrices A and b directly without using Polynomial
    # - implement in c++


if __name__ == '__main__':
    main()
