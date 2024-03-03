from envs.base_config import BaseConfig
import numpy as np

class NightmareV3Config(BaseConfig):
    device = 'cpu'
    rl_device = 'cuda'

    class env:
        dt = 0.002
        model_path = 'models/nightmare_v3/mjmodel.xml'
        num_envs = 2048
        num_obs = 66
        num_privileged_obs = 0
        num_actions = 18
        episode_length_s = 10
        send_timeouts = True
        feet_names = ['leg_1_foot', 'leg_2_foot', 'leg_3_foot', 'leg_4_foot', 'leg_5_foot', 'leg_6_foot']
        body_name = 'base_link'
        termination_names = [
            'leg_1_coxa', 'leg_1_femur', 'leg_1_tibia',
            'leg_2_coxa', 'leg_2_femur', 'leg_2_tibia',
            'leg_3_coxa', 'leg_3_femur', 'leg_3_tibia',
            'leg_4_coxa', 'leg_4_femur', 'leg_4_tibia',
            'leg_5_coxa', 'leg_5_femur', 'leg_5_tibia',
            'leg_6_coxa', 'leg_6_femur', 'leg_6_tibia',
            'body_link']
        floor_name = 'floor'
        termination_contact_force = 50.0

    class viewer:
        render = True
        record_states = True

    class control:
        p_gain = 10
        d_gain = 0.05
        default_pos = [0, np.pi / 5, 0, 
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0]
        decimation = 4
        action_scale = 0.3

    class noise:
        add_noise = False
        noise_level = 0.1
        class noise_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            gravity = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 1.0
    
    class commands:
        resampling_time = 5
        class ranges:
            max_lin_vel_x = 0.4
            max_lin_vel_y = 0.4
            max_ang_vel = 0.6
    
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.0
        clip_actions = 1.0
    
    class rewards:
        class scales:
            termination = -200.0
            tracking_lin_vel = 10.0
            tracking_ang_vel = 5.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.003
            dof_acc = -1e-8
            base_height = -10.0
            feet_air_time =  1.0
            # collision = -1.
            # feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -1.0
            default_position = -0.5
            feet_contact_forces = -0.05

        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.1
        max_contact_force = 10. # forces above this value are penalized
    
class NightmareV3ConfigPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 62 # per iteration
        max_iterations = 1000000000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt