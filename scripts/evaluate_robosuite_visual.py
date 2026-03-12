#!/usr/bin/env python3
"""
Evaluate a trained visual diffusion policy checkpoint on Robosuite environments.

Key fixes vs v1:
  - frame_skip=1  (data was recorded at control_freq=20Hz, each step = 1 control step)
  - observation history buffer for obs_horizon > 1
  - no action clipping after denorm (policy outputs correct range)
  - use DDPM full 100 steps by default for better quality
"""
import argparse
import sys
import os
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MUJOCO_GL",        "egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")

import torch
import numpy as np
from omegaconf import OmegaConf
import robosuite as suite
import h5py, json
from tqdm import tqdm

from diffusion.visual_ddpm import VisualDiffusionPolicy
from diffusion.ddpm import DiffusionPolicy
from utils.normalizer import RunningNormalizer
from env.video_recorder import VideoRecorder


# ─────────────────────────────────────────────────────────────────────────── #
# Robosuite wrapper                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class RobosuiteEnvWrapper:
    def __init__(
        self,
        env_args_str: str,
        camera_names: list,
        img_size: int = 84,
        seed: int = 0,
    ):
        self._camera_names = camera_names
        self._img_size     = img_size
        self._seed         = seed

        cfg = json.loads(env_args_str)
        rs_name   = cfg.get("env_name", "PickPlaceCan")
        rs_kwargs = cfg.get("env_kwargs", {})

        rs_kwargs["has_renderer"]           = False
        rs_kwargs["has_offscreen_renderer"] = True
        rs_kwargs["use_camera_obs"]         = True
        rs_kwargs["camera_names"]           = camera_names
        rs_kwargs["camera_heights"]         = img_size
        rs_kwargs["camera_widths"]          = img_size
        rs_kwargs["camera_depths"]          = False
        rs_kwargs["ignore_done"]            = True
        rs_kwargs["reward_shaping"]         = False  # match training
        rs_kwargs.pop("env_name", None)

        self.rs_env = suite.make(rs_name, **rs_kwargs)

    def reset(self, seed=None):
        obs_dict = self.rs_env.reset()
        return self._process(obs_dict)

    def step(self, action: np.ndarray):
        """Execute one control step (frame_skip=1, matching training data)."""
        obs_dict, reward, done, info = self.rs_env.step(action)
        return self._process(obs_dict), float(reward), done, info

    def _process(self, obs_dict: dict) -> dict:
        cam = self._camera_names[0]
        img = obs_dict[f"{cam}_image"]           # (H, W, 3) uint8
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # (3, H, W)

        state = np.concatenate([
            obs_dict["robot0_eef_pos"],      # 3
            obs_dict["robot0_eef_quat"],     # 4
            obs_dict["robot0_gripper_qpos"], # 2
        ]).astype(np.float32)                # (9,)

        return {"image": img, "state": state}

    def is_success(self):
        return bool(self.rs_env._check_success())

    def render(self):
        raw = self.rs_env._get_observations(force_update=False)
        img = raw.get(f"{self._camera_names[0]}_image")
        return img  # (H, W, 3) uint8 or None


# ─────────────────────────────────────────────────────────────────────────── #
# Rollout with obs history buffer                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def rollout(policy, env, obs_horizon, action_horizon, max_steps,
            seed, num_ddim_steps, eta, device, record_frames=False):
    obs = env.reset(seed=seed)

    # obs history deques (length = obs_horizon)
    img_buf   = deque([obs["image"]]   * obs_horizon, maxlen=obs_horizon)
    state_buf = deque([obs["state"]]   * obs_horizon, maxlen=obs_horizon)

    total_reward = 0.0
    success = False
    frames = [] if record_frames else None
    step = 0

    while step < max_steps:
        # Build batch tensors
        imgs   = np.stack(img_buf, axis=0)     # (obs_h, 3, H, W)
        states = np.concatenate(state_buf)      # (obs_h * state_dim,)

        imgs_t   = torch.from_numpy(imgs[None]).to(device)    # (1, obs_h, 3, H, W)
        states_t = torch.from_numpy(states[None]).to(device)  # (1, obs_h * state_dim)

        # Predict action chunk
        actions = policy.predict_action(
            imgs_t, states_t,
            num_ddim_steps=num_ddim_steps,
            eta=eta,
        )  # (1, action_horizon, act_dim)
        actions = actions[0].cpu().numpy()  # (action_horizon, act_dim)

        # Execute chunk (receding horizon)
        for t in range(action_horizon):
            if step >= max_steps:
                break

            obs, reward, done, info = env.step(actions[t])
            img_buf.append(obs["image"])
            state_buf.append(obs["state"])
            total_reward += reward
            step += 1

            if record_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if env.is_success():
                success = True

            if done:
                break

        if success or done:
            break

    return {"reward": total_reward, "success": success, "length": step, "frames": frames}


# ─────────────────────────────────────────────────────────────────────────── #
# Checkpoint loader                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def load_policy(checkpoint_path, cfg, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy = VisualDiffusionPolicy(
        state_dim          = cfg.dataset.state_dim,
        act_dim            = cfg.dataset.act_dim,
        action_horizon     = cfg.dataset.action_horizon,
        obs_horizon        = cfg.dataset.obs_horizon,
        encoder_type       = cfg.model.encoder_type,
        img_emb_dim        = cfg.model.img_emb_dim,
        img_fusion         = cfg.model.img_fusion,
        encoder_frozen     = cfg.model.encoder_frozen,
        encoder_pretrained = cfg.model.encoder_pretrained,
        state_emb_dim      = cfg.model.state_emb_dim,
        hidden_dim         = cfg.model.hidden_dim,
        num_layers         = cfg.model.num_layers,
        time_emb_dim       = cfg.model.time_emb_dim,
        num_train_timesteps= cfg.diffusion.num_train_timesteps,
        beta_schedule      = cfg.diffusion.beta_schedule,
        prediction_type    = cfg.diffusion.prediction_type,
        clip_sample        = cfg.diffusion.clip_sample,
        clip_sample_range  = cfg.diffusion.clip_sample_range,
        device             = device,
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval().to(device)

    if ckpt.get("state_normalizer") is not None:
        policy.state_normalizer = RunningNormalizer()
        policy.state_normalizer.load_state_dict(ckpt["state_normalizer"])
    if ckpt.get("act_normalizer") is not None:
        policy.act_normalizer = RunningNormalizer()
        policy.act_normalizer.load_state_dict(ckpt["act_normalizer"])

    print(f"[load_policy] {checkpoint_path}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return policy


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--config",          required=True)
    p.add_argument("--num_episodes",    type=int, default=10)
    p.add_argument("--max_steps",       type=int, default=400)
    p.add_argument("--num_ddim_steps",  type=int, default=100)
    p.add_argument("--eta",             type=float, default=0.0)
    p.add_argument("--video_dir",       type=str, default=None)
    p.add_argument("--max_videos",      type=int, default=None,
                   help="Max number of episode videos to save (default: all)")
    p.add_argument("--seed_offset",     type=int, default=0)
    p.add_argument("--device",          type=str, default=None)
    p.add_argument("--tb_dir",          type=str, default=None,
                   help="TensorBoard log dir. If set, logs scalars + videos.")
    p.add_argument("--tb_tag",          type=str, default=None,
                   help="Tag prefix for TensorBoard (e.g. 'obs_h2'). Defaults to obs_horizon from config.")
    p.add_argument("--tb_step",         type=int, default=None,
                   help="Global step for TensorBoard x-axis (e.g. obs_horizon value).")
    p.add_argument("--tb_max_videos",   type=int, default=4,
                   help="Max number of episode videos to log to TensorBoard.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load env_args from raw HDF5
    task = cfg.dataset.task
    raw_path = Path(f"data/robomimic/{task}_ph_raw.hdf5")
    if not raw_path.exists():
        raw_path = Path("data/robomimic/can_ph_raw.hdf5")
    with h5py.File(raw_path, "r") as f:
        env_args_str = f["data"].attrs["env_args"]

    cam = cfg.dataset.camera_names
    if isinstance(cam, str):
        cam = [cam]

    env = RobosuiteEnvWrapper(
        env_args_str = env_args_str,
        camera_names = cam,
        img_size     = cfg.dataset.img_size,
    )
    policy = load_policy(args.checkpoint, cfg, device)

    obs_horizon    = cfg.dataset.obs_horizon
    action_horizon = cfg.dataset.action_horizon

    # TensorBoard setup
    writer   = None
    tb_tag   = args.tb_tag or f"obs_h{obs_horizon}"
    tb_step  = args.tb_step if args.tb_step is not None else obs_horizon
    if args.tb_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_dir)
        print(f"[TensorBoard] logging to {args.tb_dir}  tag={tb_tag}  step={tb_step}")

    print(f"\n[Eval] {args.num_episodes} episodes  |  obs_h={obs_horizon}  "
          f"act_h={action_horizon}  ddim_steps={args.num_ddim_steps}  "
          f"frame_skip=1 (fixed)\n")

    rewards, lengths, successes = [], [], []
    videos_saved = 0
    # collect frames for TB video logging: up to tb_max_videos success + failure each
    tb_videos_success, tb_videos_fail = [], []

    pbar = tqdm(range(args.num_episodes), desc="Evaluating")
    for i in pbar:
        seed = args.seed_offset + i
        save_video = args.video_dir is not None or writer is not None

        result = rollout(
            policy=policy, env=env,
            obs_horizon=obs_horizon, action_horizon=action_horizon,
            max_steps=args.max_steps, seed=seed,
            num_ddim_steps=args.num_ddim_steps, eta=args.eta,
            device=device, record_frames=save_video,
        )
        rewards.append(result["reward"])
        lengths.append(result["length"])
        successes.append(result["success"])

        # save mp4 (respect --max_videos limit)
        if args.video_dir and result["frames"] and (args.max_videos is None or videos_saved < args.max_videos):
            vpath = Path(args.video_dir) / f"ep_{i:02d}.mp4"
            vpath.parent.mkdir(parents=True, exist_ok=True)
            rec = VideoRecorder(str(vpath), fps=20)
            for fr in result["frames"]:
                rec.add_frame(fr)
            rec.save()
            videos_saved += 1

        # collect for TensorBoard video
        if writer and result["frames"]:
            frames_np = np.stack(result["frames"])  # (T, H, W, 3) uint8
            if result["success"] and len(tb_videos_success) < args.tb_max_videos:
                tb_videos_success.append(frames_np)
            elif not result["success"] and len(tb_videos_fail) < args.tb_max_videos:
                tb_videos_fail.append(frames_np)

        pbar.set_postfix(
            sr=f"{np.mean(successes):.0%}",
            r=f"{result['reward']:.2f}",
            ok=result["success"],
        )

    success_rate = np.mean(successes)
    mean_reward  = np.mean(rewards)
    mean_length  = np.mean(lengths)

    print("\n" + "═"*50)
    print(f"  success_rate  {success_rate:.4f}  ({sum(successes)}/{len(successes)})")
    print(f"  mean_reward   {mean_reward:.4f}")
    print(f"  mean_length   {mean_length:.1f}")
    print("═"*50)
    print(f"  success_rate  {success_rate:.4f}")  # grep-friendly

    # ── TensorBoard logging ──────────────────────────────────────────────
    if writer:
        # Scalars
        writer.add_scalar(f"eval/success_rate", success_rate, tb_step)
        writer.add_scalar(f"eval/mean_reward",  mean_reward,  tb_step)
        writer.add_scalar(f"eval/mean_length",  mean_length,  tb_step)
        writer.add_scalar(f"eval/n_success",    sum(successes), tb_step)
        print(f"[TensorBoard] scalars logged at step={tb_step}")

        # Videos: (N, T, C, H, W) uint8
        def _log_videos(frames_list, tag):
            if not frames_list:
                return
            # pad all to same length
            max_T = max(f.shape[0] for f in frames_list)
            padded = []
            for f in frames_list:
                if f.shape[0] < max_T:
                    pad = np.repeat(f[-1:], max_T - f.shape[0], axis=0)
                    f = np.concatenate([f, pad], axis=0)
                # (T, H, W, 3) → (T, 3, H, W)
                padded.append(f.transpose(0, 3, 1, 2))
            vid = np.stack(padded)  # (N, T, C, H, W)
            writer.add_video(tag, vid, global_step=tb_step, fps=20)
            print(f"[TensorBoard] video logged: {tag}  shape={vid.shape}")

        _log_videos(tb_videos_success, f"{tb_tag}/success")
        _log_videos(tb_videos_fail,    f"{tb_tag}/failure")

        writer.flush()
        writer.close()
        print(f"[TensorBoard] done → open http://localhost:6006")


if __name__ == "__main__":
    main()
