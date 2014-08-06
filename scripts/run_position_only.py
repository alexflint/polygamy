import os
from fractions import Fraction
from pathlib import Path
import numdifftools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import spline
from utils import cayley, normalized, skew, skew_jacobian, essential_matrix
from polynomial import Polynomial, quadratic_form
from lie import SO3


class BezierCurve(object):
    def __init__(self, controls):
        self._controls = controls

    def evaluate(self, t):
        return spline.bezier(self._controls, t)

    def second_derivative(self, t):
        return spline.bezier_second_deriv(self._controls, t)


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


def diagify(x, k):
    x = np.atleast_2d(x)
    m, n = x.shape
    out = np.zeros((m*k, n*k), x.dtype)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i*k:i*k+k, j*k:j*k+k] = np.eye(k) * x[i, j]
    return out


def dots(*args):
    return reduce(np.dot, args)


def accel_residual(pos_controls, gravity, accel_bias,
                   timestamp, accel_reading, orientation):
    global_accel = spline.zero_offset_bezier_second_deriv(pos_controls, timestamp)
    apparent_accel = np.dot(orientation, global_accel + gravity) + accel_bias
    return apparent_accel - accel_reading


def accel_jacobian(bezier_order, timestamp, orientation):
    bezier_mat = spline.zero_offset_bezier_second_deriv_mat(timestamp, bezier_order, 3)
    return np.hstack((np.dot(orientation, bezier_mat), orientation, np.eye(3)))


def evaluate_accel_residuals(pos_controls, gravity, accel_bias,
                             accel_timestamps, accel_readings, accel_orientations):
    return np.array([accel_residual(pos_controls, gravity, accel_bias, t, accel, R)
                     for t, R, accel in zip(accel_timestamps, accel_orientations, accel_readings)])


def epipolar_residual(pos_controls, ti, tj, zi, zj, Ri, Rj):
    pi = spline.zero_offset_bezier(pos_controls, ti)
    pj = spline.zero_offset_bezier(pos_controls, tj)
    E = essential_matrix(Ri, pi, Rj, pj)
    return dots(zj, E, zi)


def epipolar_jacobian(bezier_order, ti, tj, zi, zj, Ri, Rj):
    Rrel = np.dot(Rj, Ri.T)
    zzt = np.outer(zj, zi).flatten()
    Ai = spline.zero_offset_bezier_mat(ti, bezier_order, 3)
    Aj = spline.zero_offset_bezier_mat(tj, bezier_order, 3)
    return dots(zzt, diagify(Rrel, 3), skew_jacobian(), np.dot(Ri, Aj - Ai))


def evaluate_epipolar_residuals(pos_controls, frame_timestamps, frame_orientations,
                                features, feature_mask=None):
    residuals = []
    for i, (ti, Ri) in enumerate(zip(frame_timestamps, frame_orientations)):
        for j, (tj, Rj) in enumerate(zip(frame_timestamps, frame_orientations)):
            if i != j:
                for k in range(features.shape[1]):
                    if feature_mask is None or (feature_mask[i, k] and feature_mask[i, j]):
                        zi = features[i][k]
                        zj = features[j][k]
                        residuals.append(epipolar_residual(pos_controls, ti, tj, zi, zj, Ri, Rj))
    return np.array(residuals)


def run_accel_finite_differences():
    np.random.seed(0)

    bezier_order = 4
    pos_controls = np.random.randn(bezier_order, 3)
    accel_bias = np.random.randn(3)
    gravity = np.random.randn(3)
    a = np.random.randn(3)
    R = SO3.exp(np.random.randn(3))
    t = .5

    def r(delta):
        k = bezier_order * 3
        assert len(delta) == k + 6
        return accel_residual(pos_controls + delta[:k].reshape((bezier_order, 3)),
                              gravity + delta[k:k+3],
                              accel_bias + delta[k+3:k+6],
                              t,
                              a,
                              R)

    J_numeric = numdifftools.Jacobian(r)(np.zeros(bezier_order*3+6))
    J_analytic = accel_jacobian(bezier_order, t, R)

    print '\nNumeric:'
    print J_numeric

    print '\nAnalytic:'
    print J_analytic

    np.testing.assert_array_almost_equal(J_numeric, J_analytic, decimal=8)


def run_epipolar_finite_differences():
    np.random.seed(0)

    bezier_order = 4
    pos_controls = np.random.randn(bezier_order, 3)
    ti, tj = np.random.randn(2)
    zi, zj = np.random.randn(2, 3)
    Ri = SO3.exp(np.random.randn(3))
    Rj = SO3.exp(np.random.randn(3))

    def r(delta):
        assert len(delta) == bezier_order * 3
        return epipolar_residual(pos_controls + delta.reshape((bezier_order, 3)),
                                 ti, tj, zi, zj, Ri, Rj)

    J_numeric = np.squeeze(numdifftools.Jacobian(r)(np.zeros(bezier_order*3)))
    J_analytic = epipolar_jacobian(bezier_order, ti, tj, zi, zj, Ri, Rj)

    print '\nNumeric:'
    print J_numeric

    print '\nAnalytic:'
    print J_analytic

    np.testing.assert_array_almost_equal(J_numeric, J_analytic, decimal=8)


def run_simulation():
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

    true_frame_cayleys = np.array([spline.zero_offset_bezier(true_rot_controls, t) for t in frame_times])
    true_frame_orientations = np.array(map(cayley, true_frame_cayleys))
    true_frame_positions = np.array([spline.zero_offset_bezier(true_pos_controls, t) for t in frame_times])

    true_imu_cayleys = np.array([spline.zero_offset_bezier(true_rot_controls, t) for t in imu_times])
    true_imu_orientations = np.array(map(cayley, true_imu_cayleys))

    true_gravity_magnitude = 9.8
    true_gravity = normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3)
    true_global_accels = np.array([spline.zero_offset_bezier_second_deriv(true_pos_controls, t) for t in imu_times])
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
    # Compute residuals
    #

    epipolar_residuals = evaluate_epipolar_residuals(sym_pos_controls, frame_times,
                                                     observed_frame_orientations, observed_features)
    accel_residuals = evaluate_accel_residuals(sym_pos_controls, sym_gravity, sym_accel_bias,
                                               imu_times, observed_accels, observed_imu_orientations)

    residuals = np.hstack((accel_residuals, epipolar_residuals))

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
    estimated_positions = np.array([spline.zero_offset_bezier(estimated_pos_controls, t) for t in frame_times])
    estimated_accel_bias = np.asarray(estimated_vars[accel_bias_offset:accel_bias_offset+3])
    estimated_gravity = np.asarray(estimated_vars[gravity_offset:gravity_offset+3])
    re_estimated_gravity = normalized(estimated_gravity) * true_gravity_magnitude

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
    true_ps = np.array([spline.zero_offset_bezier(true_pos_controls, t) for t in ts])
    estimated_ps = np.array([spline.zero_offset_bezier(estimated_pos_controls, t) for t in ts])

    ax.plot(true_ps[:, 0], true_ps[:, 1], true_ps[:, 2], '-b')
    ax.plot(estimated_ps[:, 0], estimated_ps[:, 1], estimated_ps[:, 2], '-r')

    plt.show()

    # TODO:
    # - get real dataset
    # - python wrappers for feature tracker
    # - python wrappers for five point algorithm
    # - build up problem matrices A and b directly without using Polynomial
    # - implement in c++


def run_simulation_nonsymbolic():
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

    true_frame_cayleys = np.array([spline.zero_offset_bezier(true_rot_controls, t) for t in frame_times])
    true_frame_orientations = np.array(map(cayley, true_frame_cayleys))
    true_frame_positions = np.array([spline.zero_offset_bezier(true_pos_controls, t) for t in frame_times])

    true_imu_cayleys = np.array([spline.zero_offset_bezier(true_rot_controls, t) for t in imu_times])
    true_imu_orientations = np.array(map(cayley, true_imu_cayleys))

    true_gravity_magnitude = 9.8
    true_gravity = normalized(np.random.rand(3)) * true_gravity_magnitude
    true_accel_bias = np.random.randn(3)
    true_global_accels = np.array([spline.zero_offset_bezier_second_deriv(true_pos_controls, t) for t in imu_times])
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
    # Compute residuals
    #

    accel_residuals = evaluate_accel_residuals(sym_pos_controls, sym_gravity, sym_accel_bias,
                                               imu_times, observed_accels, observed_imu_orientations)
    epipolar_residuals = evaluate_epipolar_residuals(sym_pos_controls, frame_times,
                                                     observed_frame_orientations, observed_features)
    residuals = np.hstack((accel_residuals, epipolar_residuals))

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
    A *= 2

    estimated_vars = np.squeeze(np.linalg.solve(A, -b))
    estimated_pos_controls = np.reshape(estimated_vars[position_offs:position_offs+(bezier_degree-1)*3], (bezier_degree-1, 3))
    estimated_positions = np.array([spline.zero_offset_bezier(estimated_pos_controls, t) for t in frame_times])
    estimated_accel_bias = np.asarray(estimated_vars[accel_bias_offset:accel_bias_offset+3])
    estimated_gravity = np.asarray(estimated_vars[gravity_offset:gravity_offset+3])
    re_estimated_gravity = normalized(estimated_gravity) * true_gravity_magnitude

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
    true_ps = np.array([spline.zero_offset_bezier(true_pos_controls, t) for t in ts])
    estimated_ps = np.array([spline.zero_offset_bezier(estimated_pos_controls, t) for t in ts])

    ax.plot(true_ps[:, 0], true_ps[:, 1], true_ps[:, 2], '-b')
    ax.plot(estimated_ps[:, 0], estimated_ps[:, 1], estimated_ps[:, 2], '-r')

    plt.show()


def run_from_data():
    #
    # Load data
    #

    path = Path('/Users/alexflint/Code/spline-initialization/out')

    frame_orientation_data = np.loadtxt(str(path / 'frame_orientations.txt'))
    frame_timestamps = frame_orientation_data[:, 0]
    frame_orientations = frame_orientation_data[:, 1:].reshape((-1, 3, 3))

    accel_data = np.loadtxt(str(path / 'accelerometer.txt'))
    accel_timestamps = accel_data[:, 0]
    accel_readings = accel_data[:, 1:]

    accel_orientation_data = np.loadtxt(str(path / 'accel_orientations.txt'))
    accel_orientations = accel_orientation_data[:, 1:].reshape((-1, 3, 3))

    feature_data = np.loadtxt(str(path / 'features.txt'))
    landmarks_ids = sorted(set(feature_data[:, 0].astype(int)))
    frame_ids = sorted(set(feature_data[:, 1].astype(int)))
    landmark_index_by_id = {idx: i for i, idx in enumerate(landmarks_ids)}
    frame_index_by_id = {idx: i for i, idx in enumerate(frame_ids)}

    assert len(accel_orientations) == len(accel_readings)
    assert len(frame_ids) == len(frame_orientations) == len(frame_timestamps)

    num_frames = len(frame_ids)
    num_landmarks = len(landmarks_ids)
    num_imu_readings = len(accel_readings)
    bezier_degree = 4

    print 'Num landmarks:', num_landmarks
    print 'Num frames:', num_frames
    print 'Num IMU readings:', num_imu_readings
    print 'Bezier curve degree:', bezier_degree

    #
    # Make feature table
    #
    features = np.ones((num_frames, num_landmarks, 2))
    feature_mask = np.zeros((num_frames, num_landmarks), bool)
    features.fill(np.nan)
    for landmark_id, frame_id, feature in zip(landmarks_ids, frame_ids, feature_data[:, 2:]):
        i = frame_index_by_id[frame_id]
        j = landmark_index_by_id[landmark_id]
        features[i, j] = feature
        feature_mask[i, j] = True

    #
    # Normalize timestamps to [0,1]
    #
    begin_time = min(np.min(accel_timestamps), np.min(frame_timestamps))
    end_time = max(np.max(accel_timestamps), np.max(frame_timestamps))
    accel_timestamps = (accel_timestamps - begin_time) / (end_time - begin_time)
    frame_timestamps = (frame_timestamps - begin_time) / (end_time - begin_time)

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

    #
    # Compute residuals
    #
    epipolar_residuals = evaluate_epipolar_residuals(sym_pos_controls, frame_timestamps,
                                                     frame_orientations, features, feature_mask)
    accel_residuals = evaluate_accel_residuals(sym_pos_controls, sym_gravity, sym_accel_bias,
                                               accel_timestamps, accel_readings, accel_orientations)

    residuals = accel_residuals + epipolar_residuals

    print '\nNum vars:', num_vars
    print 'Num residuals:', len(residuals)

    print '\nResiduals:', len(residuals)
    cost = Polynomial(num_vars)
    for r in residuals:
        cost += r*r
        print '  degree=%d, length=%d' % (r.total_degree, len(r))

    print '\nCost:'
    print '  Num terms: %d' % len(cost)
    print '  Degree: %d' % cost.total_degree

    # Solve
    A, b, k = quadratic_form(cost)
    estimated_vars = np.squeeze(np.linalg.solve(A*2, -b))
    estimated_pos_controls = np.reshape(estimated_vars[position_offs:position_offs+(bezier_degree-1)*3], (bezier_degree-1, 3))
    estimated_positions = np.array([spline.zero_offset_bezier(estimated_pos_controls, t) for t in frame_timestamps])
    estimated_accel_bias = np.asarray(estimated_vars[accel_bias_offset:accel_bias_offset+3])
    estimated_gravity = np.asarray(estimated_vars[gravity_offset:gravity_offset+3])

    print '\nEstimated:'
    print estimated_vars

    print 'Estimated accel bias:', estimated_accel_bias
    print 'Estimated gravity:', estimated_gravity
    print 'Estimated gravity magnitude:', np.linalg.norm(estimated_gravity)

    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ts = np.linspace(0, 1, 100)
    estimated_ps = np.array([spline.zero_offset_bezier(estimated_pos_controls, t) for t in ts])
    ax.plot(estimated_ps[:, 0], estimated_ps[:, 1], estimated_ps[:, 2], '-r')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

    # TODO:
    # - get real dataset
    # - python wrappers for feature tracker
    # - python wrappers for five point algorithm
    # - build up problem matrices A and b directly without using Polynomial
    # - implement in c++


if __name__ == '__main__':
    np.set_printoptions(linewidth=500)
    #run_simulation()
    #run_simulation_nonsymbolic()
    #run_from_data()
    #run_accel_finite_differences()
    run_epipolar_finite_differences()
