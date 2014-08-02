import numpy as np


def skew(m):
    m = np.asarray(m)
    return np.array([[0.,    -m[2],  m[1]],
                     [m[2],   0.,   -m[0]],
                     [-m[1],  m[0],    0.]])


def evaluate_bezier(params, t):
    """Evaluate a bezier curve at time t"""
    if len(params) == 1:
        return params[0]
    else:
        return evaluate_bezier(params[:-1], t) * (1.-t) + evaluate_bezier(params[1:], t) * t


def evaluate_bezier_deriv(params, t):
    if len(params) == 1:
        return 0.
    else:
        a = evaluate_bezier(params[:-1], t)
        b = evaluate_bezier(params[1:], t)
        aderiv = evaluate_bezier_deriv(params[:-1], t)
        bderiv = evaluate_bezier_deriv(params[1:], t)
        return aderiv*(1.-t) + bderiv*t - a + b


def evaluate_bezier_second_deriv(params, t):
    if len(params) == 1:
        return 0.
    else:
        aderiv = evaluate_bezier_deriv(params[:-1], t)
        bderiv = evaluate_bezier_deriv(params[1:], t)
        aderiv2 = evaluate_bezier_second_deriv(params[:-1], t)
        bderiv2 = evaluate_bezier_second_deriv(params[1:], t)
        return aderiv2*(1.-t) + bderiv2*t - aderiv*2. + bderiv*2.


def evaluate_zero_offset_bezier(params, t):
    if t == 0:
        # should return a numpy array for t=0, not a Polynomial
        return np.zeros_like(params[0])
    else:
        return evaluate_bezier(np.vstack((np.zeros(len(params[0])), params)), t)


def evaluate_zero_offset_bezier_second_deriv(params, t):
    return evaluate_bezier_second_deriv(np.vstack((np.zeros(len(params[0])), params)), t)


def cayley_mat(s):
    return np.eye(3) * (1. - np.dot(s, s)) + 2.*skew(s) + 2.*np.outer(s, s)


def cayley_denom(s):
    return 1. + np.dot(s, s)


def cayley(s):
    return cayley_mat(s) / cayley_denom(s)


def essential_matrix(Rrel, prel):
    return np.dot(Rrel, skew(prel))


def accel_residual(pos_controls, orient_controls, accel_bias, gravity, observed_accel, time):
    cayley = evaluate_zero_offset_bezier(orient_controls, time)
    orientation = cayley_mat(cayley)
    denom = cayley_denom(cayley)
    global_accel = evaluate_zero_offset_bezier_second_deriv(pos_controls, time)
    accel = np.dot(orientation, global_accel + gravity) + denom * accel_bias
    return accel - denom * observed_accel


def epipolar_residuals(pos_controls, orient_controls, observed_feature, observed_feature_first_frame, time):
    orientation = cayley_mat(evaluate_zero_offset_bezier(orient_controls, time))
    position = evaluate_zero_offset_bezier(pos_controls, time)
    E = essential_matrix(orientation, position)
    return np.dot(observed_feature, np.dot(E, observed_feature_first_frame))


def cost(pos_controls, orient_controls, accel_bias, gravity,
         observed_features, observed_frame_times,
         observed_accels, observed_accel_times):
    cost = 0.

    # Accel residuals
    for i in range(len(observed_accels)):
        r = accel_residual(pos_controls, orient_controls, accel_bias, gravity,
                           observed_accels[i], observed_accel_times[i])
        cost += r[0]*r[0] + r[1]*r[1] + r[2]*r[2]

    # Epipolar residuals
    for i in range(1, len(observed_frame_times)):    # loop over frames 2,3,...,N
        for j in range(observed_features.shape[1]):  # loop over landmarks 1..K
            r = epipolar_residuals(pos_controls, orient_controls,
                                   observed_features[i,j],  # landmark j in frame i
                                   observed_features[0,j],  # landmark j in frame 1
                                   observed_frame_times[i])
            cost += r*r

    return cost


def main():
    num_frames, num_landmarks, num_imu_readings = np.loadtxt('problem_size.txt')
    observed_features = np.loadtxt('feature_measurements.txt').reshape((num_frames, num_landmarks, 3))
    observed_accels = np.loadtxt('accel_measurements.txt')
    frame_times = np.loadtxt('frame_times.txt')
    accel_times = np.loadtxt('accel_times.txt')

    true_pos_controls = np.loadtxt('true_pos_controls.txt')
    true_orient_controls = np.loadtxt('true_orient_controls.txt')
    true_accel_bias = np.loadtxt('true_accel_bias.txt')
    true_gravity = np.loadtxt('true_gravity.txt')

    print cost(true_pos_controls, true_orient_controls, true_accel_bias, true_gravity,
               observed_features, frame_times,
               observed_accels, accel_times)

if __name__ == '__main__':
    main()
