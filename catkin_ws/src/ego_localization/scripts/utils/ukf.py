import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver


class BasicUKF(UKF):
    def __init__(self, dt=0.1, x_init=np.ones(7)) -> None:
        self.dt = dt

        self.yaw_threshold = 5e-2           # Yaw threshold to chose model
        self.L = 2.7                        # Distance between front and rear axes (kalipo)
        self.L = 2.588                      # (roboracer)
        self.steering_wheel_ratio = 16.82439       # Ratio of steering angle to wheel angle

        self.sigma_points = MerweScaledSigmaPoints(n=12, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UKF(dim_x=12, dim_z=2, dt=self.dt, fx=self.f_ca, hx=self.h_odometry, points=self.sigma_points) # 12 [x, x_dot, x_dot_dot;
                                                                                                                  #     y, y_dot, y_dot_dot;
                                                                                                                  #     roll, roll_dot;
                                                                                                                  #     pitch, pitch_dot;
                                                                                                                  #     yaw, yaw_dot];

        # Set the process noise
        std_f_xy = 0.5
        std_f_roll = 0.3
        std_f_pitch = 0.3
        std_f_yaw = 0.6


        # Build the process noise matrix
        self.ukf.Q = np.zeros((12, 12))
        self.ukf.Q[0:3, 0:3] = Q_discrete_white_noise(dim=3, dt=self.dt, var=std_f_xy**2)       # Noise for x, y, z
        self.ukf.Q[3:6, 3:6] = Q_discrete_white_noise(dim=3, dt=self.dt, var=std_f_xy**2)       # Noise for yaw, yaw_dot
        self.ukf.Q[6:8, 6:8] = Q_discrete_white_noise(dim=2, dt=self.dt, var=std_f_roll**2)     # Noise for roll, roll_dot
        self.ukf.Q[8:10, 8:10] = Q_discrete_white_noise(dim=2, dt=self.dt, var=std_f_pitch**2)  # Noise for pitch, pitch_dot
        self.ukf.Q[10:12, 10:12] = Q_discrete_white_noise(dim=2, dt=self.dt, var=std_f_yaw**2)  # Noise for yaw, yaw_dot

        # Set the measurement noise
        std_r_v = 0.5
        std_r_delta = 0.2
        self.R_odometry = np.diag([std_r_v**2, std_r_delta**2])

        std_r_gps = 0.01  # GPS noise
        self.R_gps = np.diag([std_r_gps**2, std_r_gps**2])  # GPS measurement noise for [x, y]

        # Set the initial state
        self.ukf.x = x_init # initial state
        self.ukf.P = np.diag([30**2, 3**2, 1**2,    # x, x_dot, x_dot_dot
                              30**2, 3**2, 1**2,    # y, y_dot, y_dot_dot
                              0.1**2, 1**2,         # roll, roll_dot, 
                              0.1**2, 1**2,         # pitch, pitch_dot
                              0.1**2, 1**2])        # yaw, yaw_dot
        
    def f_ca(self, x: np.ndarray, dt: float, u: np.ndarray = None) -> np.ndarray:
        """
        Process model using a constant-acceleration approach, driven by
        current speed and steering angle inputs, assuming a bicycle model for yaw dynamics.

        State: [x, x_dot, x_ddot, y, y_dot, y_ddot, roll, roll_dot, pitch, pitch_dot, yaw, yaw_dot]
        Control: [speed_cmd, steering_angle_cmd]

        :param x: Current state vector
        :param dt: Time step in seconds
        :param u: Control input vector [speed_cmd, steering_angle_cmd] (or None)
        :return: Updated state vector
        """
        # Unpack state variables
        pos_x, vel_x, acc_x, pos_y, vel_y, acc_y, \
        roll, roll_dot, pitch, pitch_dot, yaw, yaw_dot = x.copy() # Work with a copy to avoid modifying original x

        # Get control inputs
        speed_cmd = 0.0
        steering_angle_cmd = 0.0

        if u is not None and len(u) >= 2:
            speed_cmd = u[0]
            steering_angle_cmd = u[1]

        # --- Calculate intermediate values based on control inputs and current state ---

        # 1. Yaw Rate (omega_z or yaw_dot_predicted) from Bicycle Model
        # Omega = v * tan(steering_angle) / L
        # This is the commanded yaw rate implied by the inputs.
        # Handle division by zero for steering_angle_cmd (or small values)
        if abs(steering_angle_cmd) < 1e-4: # Effectively straight
            yaw_rate_model = 0.0
        else:
            yaw_rate_model = speed_cmd * np.tan(steering_angle_cmd) / self.L

        # We will use this `yaw_rate_model` to update the state's `yaw_dot`
        # and drive the yaw propagation.

        # 2. Predicted Body Velocities based on commanded speed
        # Assuming the commanded speed is the new longitudinal speed of the vehicle.
        # This is then projected into the global frame using the current yaw.
        # We assume the commanded speed is achieved instantly or is the actual current speed.
        # A more complex model might integrate acceleration to reach the commanded speed.
        # For a simple kinematic model, setting current speed is common.
        vel_x_body = speed_cmd
        vel_y_body = 0.0 # Assuming no lateral slip in the bicycle model for body velocity


        # --- Propagate State ---
        x_out = x.copy() # Initialize the output state with a copy of current state

        # 1. Update Yaw and Yaw Rate
        # Propagate yaw based on the *current* yaw and the *model-derived* yaw rate
        # For yaw_dot, we update it to the `yaw_rate_model` directly, as this is driven by input.
        # This implies your control directly sets the yaw rate.
        yaw_next = yaw + yaw_rate_model * dt
        x_out[10] = self.normalize_angle(yaw_next) # Store normalized yaw
        x_out[11] = yaw_rate_model # Update yaw_dot state to the value derived from input


        # 2. Update Global Positions and Velocities (x, x_dot, y, y_dot)
        # These are derived from the commanded speed and the updated yaw.
        # We need to project the commanded speed (which is longitudinal) into the global frame.
        # The 'x_dot', 'y_dot' states are global velocities.

        # Calculate new global velocities based on commanded speed and future yaw
        # This assumes the commanded speed acts over dt and changes current velocities.
        # Or you can assume commanded speed *is* the new velocity magnitude.
        # Let's assume `speed_cmd` is the velocity magnitude over this `dt`.
        vel_x_new_global = speed_cmd * np.cos(yaw_next)
        vel_y_new_global = speed_cmd * np.sin(yaw_next)

        # Calculate accelerations based on the change in global velocities (implicit)
        # This is where your x_ddot, y_ddot states are populated as a *consequence* of the motion.
        # acc_x = (vel_x_new_global - vel_x) / dt
        # acc_y = (vel_y_new_global - vel_y) / dt

        # Update positions using trapezoidal integration (current vel + new vel)
        x_out[0] = pos_x + 0.5 * (vel_x + vel_x_new_global) * dt # x
        x_out[3] = pos_y + 0.5 * (vel_y + vel_y_new_global) * dt # y

        # Update global velocity states
        x_out[1] = vel_x_new_global # x_dot
        x_out[4] = vel_y_new_global # y_dot

        # Update acceleration states based on change in velocity
        # If dt is 0 or very small, this can cause division by zero or large numbers.
        # Only update if dt is significant.
        if dt > 1e-6:
            x_out[2] = (vel_x_new_global - vel_x) / dt # x_ddot
            x_out[5] = (vel_y_new_global - vel_y) / dt # y_ddot
        else:
            x_out[2] = 0.0
            x_out[5] = 0.0


        # 3. Update Roll and Pitch and their rates
        # Assuming roll, pitch, roll_dot, pitch_dot are *not* directly controlled by speed/steering.
        # They propagate based on their current rates (Constant Velocity model for these).
        # If you have specific controls for these (e.g., from an IMU or suspension system),
        # you would add them to the `u` input and modify these lines.
        x_out[6] = roll + roll_dot * dt # roll
        x_out[7] = roll_dot             # roll_dot (remains constant)
        x_out[8] = pitch + pitch_dot * dt # pitch
        x_out[9] = pitch_dot            # pitch_dot (remains constant)


        return x_out
        
    # def f_ca(self, x: np.ndarray, dt: float) -> np.ndarray:
    #     """
    #     Constant-acceleration bicycle-style motion model, still using the original
    #     12-element state …

    #         [x,  x_dot (=v*cos ψ),  x_ddot (=a_long),
    #         y,  y_dot (=v*sin ψ),  y_ddot (unused, keep 0),
    #         roll, roll_dot,
    #         pitch, pitch_dot,
    #         yaw, yaw_dot]

    #     Positions are global, velocities are global, acceleration is assumed to be
    #     *longitudinal* (along the body x-axis).  The heading ψ (=yaw) steers the
    #     velocity vector each step so the XY path follows the turning car.
    #     """

    #     v_body = np.hypot(x[1], x[4])  # speed magnitude (global velocity norm)

    #     # Predict next yaw (heading)
    #     yaw_next = x[10] + x[11] * dt

    #     # Predict next body-frame velocity magnitude by adding longitudinal acceleration
    #     v_body_next = v_body + np.hypot(x[2], x[5]) * dt

    #     # Update positions with trapezoidal integration projected into global frame
    #     x_out = x.copy()
    #     x_out[0] += 0.5 * (x[1] + v_body_next * np.cos(yaw_next)) * dt  # x
    #     x_out[1] = v_body_next * np.cos(yaw_next)                       # x_dot
    #     x_out[2] += 0                                                   # x_dot_dot
    #     x_out[3] += 0.5 * (x[4] + v_body_next * np.sin(yaw_next)) * dt  # y
    #     x_out[4] = v_body_next * np.sin(yaw_next)                       # y_dot
    #     x_out[5] += 0                                                   # y_dot_dot
    #     x_out[6] += x[7] * dt                                           # roll
    #     x_out[7] += 0                                                   # roll_dot
    #     x_out[8] += x[9] * dt                                           # pitch
    #     x_out[9] += 0                                                   # pitch_dot
    #     x_out[10] = yaw_next                                            # yaw
    #     x_out[11] += 0                                                  # yaw_dot

    #     return x_out

    def f_same(self, x: np.array, dt: float) -> np.array:
        """
        State transition function that does not change the state.
        This is used when no motion occurs.
        x: state vector [x, x_dot, x_dot_dot,
                         y, y_dot, y_dot_dot,
                         roll, roll_dot,
                         pitch, pitch_dot,
                         yaw, yaw_dot]
        dt: time step
        """
        return x.copy()

    def h_gps(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts [x, y] based on the current state.
        x: state vector [x, x_dot, x_dot_dot, 
                         y, y_dot, y_dot_dot, 
                         roll, roll_dot, 
                         pitch, pitch_dot, 
                         yaw, yaw_dot]"""
        # Extract position elements
        x_pos = x[0]
        y_pos = x[3]

        return np.array([x_pos, y_pos])
    
    def h_odometry(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts [velocity_body, steering_angle] based on the current state.
        x: state vector [x, x_dot, x_dot_dot, 
                         y, y_dot, y_dot_dot, 
                         roll, roll_dot, 
                         pitch, pitch_dot, 
                         yaw, yaw_dot]"""
        # Extract state elements
        x_dot = x[1]
        y_dot = x[4]
        yaw = x[10]
        yaw_rate = x[11]

        # Compute body-frame velocity (project global velocity into body frame)
        # velocity_body = np.sqrt(x_dot**2 + y_dot**2) + 1e-6 
        velocity_body = np.cos(yaw) * x_dot + np.sin(yaw) * y_dot + 1e-6

        # Compute steering angle from yaw rate and velocity
        if velocity_body < 1e-3 or abs(yaw_rate) < 1e-4:
            wheel_angle = 0.0
        else:
            wheel_angle = np.arctan2(yaw_rate * self.L, velocity_body)

        # Compute steering angle in radians
        steering_angle = wheel_angle / self.steering_wheel_ratio  # Convert wheel angle to steering angle

        return np.array([velocity_body, steering_angle], dtype=np.float64)
    
    def residual_z_odometry(self, z_pred: np.array, z_actual: np.array) -> np.array:
        """
        Compute the residual (difference) for the measurement vector z (z_pred - z_actual).
        """
        # Residual for the measurement vector
        # z_pred: predicted measurement [velocity_body, steering_angle]
        # z_actual: actual measurement [velocity_body, steering_angle]
        
        # Normalize the steering angle to be in the range [-pi, pi]
        steering_angle_pred = self.normalize_angle(z_pred[1])
        steering_angle_actual = self.normalize_angle(z_actual[1])
        
        return np.array([z_pred[0] - z_actual[0], 
                         self.normalize_angle(steering_angle_pred - steering_angle_actual)])

    def predict_ukf(self, dt=-1.0, **predict_args) -> None:
        if dt == -1.0:
            dt = self.dt
        self.ukf.predict(dt=dt, **predict_args)

    def update_ukf(self, z: np.array, **update_args) -> None:
        self.ukf.update(z, **update_args)

    def normalize_angle(self, x: float) -> float:
        # Normalize the angle to be in the range [-pi, pi]
        x = x % (2 * np.pi)
        if x > np.pi:
            x -= 2 * np.pi
        return x
    
    # def normalize_angle(self, x: float) -> float:
    #     # Normalize the angle to be in the range [-pi/2, pi/2]
    #     x = x % np.pi
    #     if x > np.pi / 2:
    #         x -= np.pi
    #     return x


if __name__ == "__main__":
    
    # Test the BasicUKF class
    from numpy.random import randn
    from tqdm import tqdm
    from filterpy.stats import plot_covariance_ellipse
    
    # Create an instance of the BasicUKF class
    dt = 0.1
    ukf = BasicUKF(dt=dt, x_init=np.ones(12))

    # Generate some random measurements
    n_measurements = 150

    # Generate velocity profile: 0 m/s until t = 2 s, then 0.2 m/s with small noise
    velocity = np.zeros(n_measurements)
    velocity[20:30] = np.linspace(0, 1, 10)
    velocity[30:60] = 1
    velocity[60:100] = np.linspace(1, 0, 40)
    velocity[100:110] = np.linspace(0, 3, 10)  # Start accelerating again
    velocity[110:140] = 3  # Hold speed
    velocity[140:150] = np.linspace(3, 0, 10)
    # velocity[:] = 0.33  # Set a constant velocity for the rest of the measurements
    velocity += np.random.normal(0, 0.05, n_measurements)           # Gaussian noise σ = 0.01 m/s

    # Generate steering angle profile: 0 rad until t = 3 s, then 0.1 rad with small noise
    steering_angle = np.zeros(n_measurements)
    # First turn right (positive angle)
    # steering_angle[30:40] = np.linspace(0, 0.2, 10)                 # Start left turn
    # steering_angle[40:70] = 0.3                                     # Hold left turn
    # steering_angle[70:100] = np.linspace(0.2, 0, 30)                # Return to straight

    # # Now turn left (negative angle)
    # steering_angle[100:110] = np.linspace(0, -0.2, 10)              # Start right turn
    # steering_angle[110:140] = -0.1                                  # Hold right turn
    # steering_angle[140:150] = np.linspace(-0.2, 0, 10)              # Return to straight
    # steering_angle[:] = 0.02  # Set a constant steering angle for the rest of the measurements
    steering_angle += np.random.normal(0, 0.01, n_measurements)    # Gaussian noise σ = 0.01 rad

    # Create the measurements
    zs = np.column_stack((velocity, steering_angle))  # [velocity, steering_angle]

    gps = np.zeros((n_measurements, 2))  # GPS measurements
    # Simulate GPS measurements for a circular path
    theta = np.linspace(0, 2*np.pi, n_measurements)
    radius = 1.4  # Radius of the circular path
    gps[:, 0] = -radius + radius * np.cos(theta)  # X position
    gps[:, 1] = radius * np.sin(theta)  # Y position
    gps += np.random.normal(0, 0.01, (n_measurements, 2))  # Add some noise to the GPS measurements
    zs_gps = np.column_stack((gps[:,0], gps[:,1]))  # [velocity, steering_angle, gps_x, gps_y]

    x_states = []
    p_states = []
    predictions = []


    # Make predictions and updates
    for i in tqdm(range(len(zs)), desc='Processing measurements'):
        z = zs[i]
        ukf.predict_ukf(fx=ukf.f_ca, dt=dt) # process noise
        predictions.append(ukf.ukf.x.copy())  # Store the predicted state

        # if i % 10 == 0:  # Update with GPS every 10 steps
        if i < 5:  # Update with GPS every 5 steps for faster convergence
            ukf.update_ukf(zs_gps[i], R=ukf.R_gps, hx=ukf.h_gps)
            ukf.predict_ukf(fx=ukf.f_same, dt=0) # process noise

        # ukf.residual_z = ukf.residual_z_odometry  # Set the residual function for odometry update
        ukf.update_ukf(z, R=ukf.R_odometry, hx=ukf.h_odometry)  # measurement update

        x_states.append(ukf.ukf.x)
        p_states.append(ukf.ukf.P)

    # Save an image of the followed path
    import matplotlib.pyplot as plt

    # Plot X and Y position of the estimated path
    x_path = [x[0] for x in x_states]
    y_path = [x[3] for x in x_states]
    x_predictions = [p[0] for p in predictions]
    y_predictions = [p[3] for p in predictions]
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(x_path, label='X Position', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('X Position (m)')
    plt.title('X Position Tracking')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y_path, label='Y Position', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position Tracking')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/data/output/figures/ukf_xy_tracking.pdf')

    ellipse_step = 20
    plt.figure(figsize=(8, 8))
    plt.plot(x_path, y_path, label='Estimated Path', color='blue')

    for i in range(len(x_path)):
        if i % ellipse_step == 0:
            # Note: ukf.x is your current state vector, but you want the history at step i
            # Similarly for ukf.P, you should access covariance at step i
            # Assuming you have lists or arrays `states_history` and `covariances_history`:
            
            mean_pos = (x_states[i][0], x_states[i][3])  # x and y positions
            # Retrieve the covariance matrix for the current state X, Y
            cov_pos = np.array([[p_states[i][0, 0], p_states[i][0, 3]],
                                [p_states[i][3, 0], p_states[i][3, 3]]])
                        
            plot_covariance_ellipse(mean_pos, cov_pos,
                                    std=1, facecolor='b', alpha=0.6)

    plt.scatter(x_predictions, y_predictions, s=5, label='Predictions', color='orange', edgecolor='black')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Estimated Path in XY Plane')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.savefig('/data/output/figures/ukf_path_xy.pdf', dpi=300)

    # Plot velocity body estimate
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    v_body = [np.hypot(x[1], x[4]) for x in x_states]
    plt.plot(v_body, label='Estimated Velocity Body', color='blue')
    plt.scatter(range(len(velocity)), velocity, label='Measured Velocity', color='red', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Velocity Body (m/s)')
    plt.title('Velocity Body Tracking')
    plt.legend()
    plt.subplot(2, 1, 2)
    acce_body = np.hypot([x[2] for x in x_states], [x[5] for x in x_states])
    plt.plot(acce_body, label='Estimated Acceleration Body', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration Body (m/s²)')
    plt.title('Acceleration Body Tracking')
    # yaw = [ukf.normalize_angle(x[10]) for x in x_states]
    # plt.plot(yaw, label='Estimated Yaw', color='blue')
    # plt.xlabel('Time Step')
    # plt.ylabel('Yaw (rad)')
    # plt.title('Yaw Tracking')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/data/output/figures/ukf_velocity_body_tracking.pdf', dpi=300)


    # Print final state with its variance
    print("Final state:")
    print(f'X: {ukf.ukf.x[0]:.2f}, var: {ukf.ukf.P[0, 0]:.2f}')
    print(f'X_dot: {ukf.ukf.x[1]:.2f}, var: {ukf.ukf.P[1, 1]:.2f}')
    print(f'X_dot_dot: {ukf.ukf.x[2]:.2f}, var: {ukf.ukf.P[2, 2]:.2f}')
    print(f'Y: {ukf.ukf.x[3]:.2f}, var: {ukf.ukf.P[3, 3]:.2f}')
    print(f'Y_dot: {ukf.ukf.x[4]:.2f}, var: {ukf.ukf.P[4, 4]:.2f}')
    print(f'Y_dot_dot: {ukf.ukf.x[5]:.2f}, var: {ukf.ukf.P[5, 5]:.2f}')
    print(f'Roll: {ukf.ukf.x[6]:.2f}, var: {ukf.ukf.P[6, 6]:.2f}')
    print(f'Roll_dot: {ukf.ukf.x[7]:.2f}, var: {ukf.ukf.P[7, 7]:.2f}')
    print(f'Pitch: {ukf.ukf.x[8]:.2f}, var: {ukf.ukf.P[8, 8]:.2f}')
    print(f'Pitch_dot: {ukf.ukf.x[9]:.2f}, var: {ukf.ukf.P[9, 9]:.2f}')
    print(f'Yaw: {ukf.ukf.x[10]:.2f}, var: {ukf.ukf.P[10, 10]:.2f}')
    print(f'Yaw_dot: {ukf.ukf.x[11]:.2f}, var: {ukf.ukf.P[11, 11]:.2f}')
    