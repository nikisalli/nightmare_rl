from rsl_rl.runners import OnPolicyRunner
import datetime
from envs.nightmare_v3_config import NightmareV3ConfigPPO, NightmareV3Config
from envs.nightmare_v3_env import NightmareV3Env
from envs.helpers import class_to_dict, get_load_path

# seconds_minutes_hours_day_month_year
date_today = datetime.datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
log_dir = f"logs/nightmare_v3/{date_today}/"
log_root = "logs/nightmare_v3/"
print(f"Logging to {log_dir}")

cfg = NightmareV3Config()
train_cfg = NightmareV3ConfigPPO()

train_cfg_dict = class_to_dict(train_cfg)

env = NightmareV3Env(cfg)
runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device=cfg.rl_device)

resume = train_cfg.runner.resume
load_run = train_cfg.runner.load_run
checkpoint = train_cfg.runner.checkpoint

if resume:
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path)

runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)