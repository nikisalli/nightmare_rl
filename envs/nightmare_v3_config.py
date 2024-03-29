from envs.base_config import BaseConfig
import numpy as np

class NightmareV3Config(BaseConfig):
    device = 'cpu'
    rl_device = 'cuda'

    class env:
        model_path = 'models/nightmare_v3/mjmodel.xml'
        num_envs = 8192
        num_obs = 66
        num_privileged_obs = 0
        num_actions = 18
        episode_length_s = 20
        send_timeouts = True
        body_name = 'base_link'
        # 0 do nothing, 1 penalize on contact, 2 terminate on contact
        tibia_contact_mode = 1
        tibia_max_contact_force = 2.0
        body_contact_mode = 1
        body_max_contact_force = 2.0
        termination_contact_force = 160.0
    
    # class oscillators:
    #     a = 1.0
    #     b = 1.0
    #     mu = 1.0
    #     max_freq = 8.0
    #     min_freq = 4.0

    class viewer:
        render = True
        record_states = True

    class control:
        p_gain = 20
        # d_gain = 0.05
        # action_rate_limit = 0.08
        default_pos = [0, np.pi / 5, 0, 
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0,
                       0, np.pi / 5, 0]
        decimation = 2
        action_scale = 0.2

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
        resampling_time = 10
        class ranges:
            max_lin_vel_x = 0.5
            max_lin_vel_y = 0.5
            max_ang_vel = 0.8
    
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
            tracking_lin_vel = 8.
            tracking_ang_vel = 6.
            dof_acc = -2.5e-5
            action_rate = -0.02
            body_contact_forces = -5 # -0.
            default_position = -0.01
            orientation = -5

            lin_vel_z = 0 # -2.0
            ang_vel_xy = 0 # -5
            feet_air_time = 0 # -4.0
            torques = 0 # -0.00001
            base_height = 0 # -2000.0
            feet_contact_forces = 0 # -0.05
            dof_vel = 0 # -0.001
            stand_still = 0 # -1.0
            collision = 0 # -1.
            feet_stumble = 0 # -0.0 

        tracking_sigma = 0.008 # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.1
        max_contact_force = 10. # forces above this value are penalized
    
class NightmareV3ConfigPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [54, 42, 30] # [512, 256, 128]
        critic_hidden_dims = [54, 42, 30] # [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        # only for 'ActorCriticODE':
        # dt = NightmareV3Config.env.dt * NightmareV3Config.control.decimation

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0015
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic' # 'ActorCriticRecurrent' # # 'ActorCriticODE'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 80 # per iteration
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