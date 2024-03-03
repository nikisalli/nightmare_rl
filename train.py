from rsl_rl.runners import OnPolicyRunner
import datetime
from envs.nightmare_v3_config import NightmareV3ConfigPPO, NightmareV3Config
from envs.nightmare_v3_env import NightmareV3Env
from envs.helpers import class_to_dict, get_load_path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resume", action="store_true", help="Resume training from the last checkpoint", default=False, dest="resume")
parser.add_argument("-v", "--render", action="store_true", help="Render the environment", default=False, dest="render")
parser.add_argument("-n", "--num_threads", type=int, help="Number of threads to use", default=1, dest="num_threads")
parser.add_argument("-e", "--envs", type=int, help="Number of environments to use", default=2048, dest="num_envs")

args = parser.parse_args()
resume = args.resume
render = args.render
num_threads = args.num_threads

print(f"Resume: {resume}, Render: {render}, Num threads: {num_threads}")

# seconds_minutes_hours_day_month_year
date_today = datetime.datetime.now()
log_dir = f"logs/nightmare_v3/{date_today}/"
log_root = "logs/nightmare_v3/"
print(f"Logging to {log_dir}")

cfg = NightmareV3Config()
train_cfg = NightmareV3ConfigPPO()

cfg.viewer.render = render
cfg.env.num_envs = args.num_envs

train_cfg.runner.resume = resume

train_cfg_dict = class_to_dict(train_cfg)

env = NightmareV3Env(cfg, log_dir=log_dir, num_threads=num_threads)
runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device=cfg.rl_device)

resume = train_cfg.runner.resume
load_run = train_cfg.runner.load_run
checkpoint = train_cfg.runner.checkpoint

if resume:
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path)

runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)