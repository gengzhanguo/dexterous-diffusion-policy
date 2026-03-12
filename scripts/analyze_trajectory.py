#!/usr/bin/env python3
"""
轨迹分析脚本：重跑 eval 同时记录 eef_pos / gripper / reward，
对比成功 vs 失败 episode 的轨迹差异。
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
import h5py, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from diffusion.visual_ddpm import VisualDiffusionPolicy
from utils.normalizer import RunningNormalizer
from scripts.evaluate_robosuite_visual import RobosuiteEnvWrapper, load_policy


# ─────────────────────────────────────────────────────────────────────────── #
# Rollout with full trajectory recording                                       #
# ─────────────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def rollout_with_traj(policy, env, obs_horizon, action_horizon,
                      max_steps, seed, num_ddim_steps, eta, device):
    obs = env.reset(seed=seed)

    img_buf   = deque([obs["image"]] * obs_horizon, maxlen=obs_horizon)
    state_buf = deque([obs["state"]] * obs_horizon, maxlen=obs_horizon)

    total_reward = 0.0
    success = False
    step = 0

    # trajectory records
    eef_pos_list   = []   # (x, y, z) 每步
    gripper_list   = []   # gripper_qpos[0] 每步
    reward_list    = []

    # 记录初始状态
    eef_pos_list.append(obs["state"][:3].copy())
    gripper_list.append(obs["state"][7])

    while step < max_steps:
        imgs   = np.stack(img_buf, axis=0)
        states = np.concatenate(state_buf)
        imgs_t   = torch.from_numpy(imgs[None]).to(device)
        states_t = torch.from_numpy(states[None]).to(device)

        actions = policy.predict_action(
            imgs_t, states_t,
            num_ddim_steps=num_ddim_steps,
            eta=eta,
        )[0].cpu().numpy()

        for t in range(action_horizon):
            if step >= max_steps:
                break
            obs, reward, done, info = env.step(actions[t])
            img_buf.append(obs["image"])
            state_buf.append(obs["state"])
            total_reward += reward
            step += 1

            eef_pos_list.append(obs["state"][:3].copy())
            gripper_list.append(obs["state"][7])
            reward_list.append(reward)

            if env.is_success():
                success = True
            if done:
                break

        if success or done:
            break

    return {
        "reward":   total_reward,
        "success":  success,
        "length":   step,
        "eef_pos":  np.array(eef_pos_list),    # (T, 3)
        "gripper":  np.array(gripper_list),     # (T,)
        "rewards":  np.array(reward_list),      # (T-1,)
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Plotting                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def plot_trajectories(results, save_path):
    n = len(results)
    success_idx = [i for i, r in enumerate(results) if r["success"]]
    fail_idx    = [i for i, r in enumerate(results) if not r["success"]]

    print(f"\n成功集: {[f'ep_{i:02d}' for i in success_idx]}")
    print(f"失败集: {[f'ep_{i:02d}' for i in fail_idx]}")

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"can_small_3k best.pt 轨迹分析  (成功: {len(success_idx)}/{n})", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    axes_labels = [
        (gs[0, 0], "eef_z (height)", lambda r: r["eef_pos"][:, 2]),
        (gs[0, 1], "eef_x",          lambda r: r["eef_pos"][:, 0]),
        (gs[1, 0], "eef_y",          lambda r: r["eef_pos"][:, 1]),
        (gs[1, 1], "gripper qpos",   lambda r: r["gripper"]),
        (gs[2, 0], "eef XY (top view)", None),
        (gs[2, 1], "eef_z vs eef_x (side)", None),
    ]

    color_s = "#2ecc71"  # green for success
    color_f = "#e74c3c"  # red for fail

    for spec, title, fn in axes_labels[:4]:
        ax = fig.add_subplot(spec)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        for i in fail_idx:
            y = fn(results[i])
            ax.plot(y, color=color_f, alpha=0.4, linewidth=1, label="fail" if i == fail_idx[0] else "")
        for i in success_idx:
            y = fn(results[i])
            ax.plot(y, color=color_s, linewidth=2, label=f"ep_{i:02d} ✓")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 高亮 success episode 最大/最小
        if success_idx and fn is not None:
            for i in success_idx:
                y = fn(results[i])
                ax.axhline(y=y.max(), color=color_s, linestyle="--", alpha=0.4, linewidth=0.8)

    # XY 俯视
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xy.set_title("eef XY top view", fontsize=11)
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    for i in fail_idx:
        xy = results[i]["eef_pos"]
        ax_xy.plot(xy[:, 0], xy[:, 1], color=color_f, alpha=0.3, linewidth=1)
        ax_xy.plot(xy[0, 0], xy[0, 1], "o", color=color_f, markersize=4)
    for i in success_idx:
        xy = results[i]["eef_pos"]
        ax_xy.plot(xy[:, 0], xy[:, 1], color=color_s, linewidth=2)
        ax_xy.plot(xy[0, 0], xy[0, 1], "o", color=color_s, markersize=6)
        ax_xy.plot(xy[-1, 0], xy[-1, 1], "*", color=color_s, markersize=10)
    ax_xy.grid(alpha=0.3)

    # XZ 侧视
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_xz.set_title("eef XZ side view (height)", fontsize=11)
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    for i in fail_idx:
        xz = results[i]["eef_pos"]
        ax_xz.plot(xz[:, 0], xz[:, 2], color=color_f, alpha=0.3, linewidth=1)
    for i in success_idx:
        xz = results[i]["eef_pos"]
        ax_xz.plot(xz[:, 0], xz[:, 2], color=color_s, linewidth=2, label=f"ep_{i:02d} ✓")
        ax_xz.plot(xz[-1, 0], xz[-1, 2], "*", color=color_s, markersize=10)
    ax_xz.legend(fontsize=8)
    ax_xz.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────── #
# Stats summary                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def print_stats(results):
    success_idx = [i for i, r in enumerate(results) if r["success"]]
    fail_idx    = [i for i, r in enumerate(results) if not r["success"]]

    print("\n" + "═"*55)
    print("  轨迹统计对比")
    print("═"*55)

    def stats(arr):
        return f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}"

    for label, idxs in [("✅ 成功", success_idx), ("❌ 失败", fail_idx)]:
        if not idxs:
            continue
        print(f"\n{label} ({len(idxs)} 集):")
        all_z    = np.concatenate([results[i]["eef_pos"][:, 2] for i in idxs])
        all_zmax = [results[i]["eef_pos"][:, 2].max() for i in idxs]
        all_len  = [results[i]["length"] for i in idxs]
        print(f"  eef_z:        {stats(all_z)}")
        print(f"  eef_z 峰值:   {np.mean(all_zmax):.4f}  (各集: {[f'{z:.4f}' for z in all_zmax]})")
        print(f"  episode 长度: mean={np.mean(all_len):.1f}  各集: {all_len}")

    # 具体看 z 峰值差距
    if success_idx and fail_idx:
        suc_zmax  = np.mean([results[i]["eef_pos"][:, 2].max() for i in success_idx])
        fail_zmax = np.mean([results[i]["eef_pos"][:, 2].max() for i in fail_idx])
        print(f"\n📊 eef_z 峰值差异: 成功 {suc_zmax:.4f}  vs  失败 {fail_zmax:.4f}  (Δ={suc_zmax-fail_zmax:+.4f})")

    print("═"*55)


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="checkpoints/can_small_3k/best.pt")
    p.add_argument("--config",         default="configs/can_small_v1.yaml")
    p.add_argument("--num_episodes",   type=int, default=10)
    p.add_argument("--max_steps",      type=int, default=400)
    p.add_argument("--num_ddim_steps", type=int, default=100)
    p.add_argument("--out",            default="logs/traj_analysis.png")
    p.add_argument("--device",         default=None)
    args = p.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    raw_path = Path("data/robomimic/can_ph_raw.hdf5")
    with h5py.File(raw_path, "r") as f:
        env_args_str = f["data"].attrs["env_args"]

    cam = cfg.dataset.camera_names
    if isinstance(cam, str):
        cam = [cam]

    env    = RobosuiteEnvWrapper(env_args_str=env_args_str, camera_names=cam, img_size=cfg.dataset.img_size)
    policy = load_policy(args.checkpoint, cfg, device)

    obs_horizon    = cfg.dataset.obs_horizon
    action_horizon = cfg.dataset.action_horizon

    print(f"\n[分析] {args.num_episodes} 集  |  obs_h={obs_horizon}  act_h={action_horizon}")

    results = []
    for i in range(args.num_episodes):
        print(f"  ep_{i:02d} ...", end="", flush=True)
        r = rollout_with_traj(
            policy=policy, env=env,
            obs_horizon=obs_horizon, action_horizon=action_horizon,
            max_steps=args.max_steps, seed=i,
            num_ddim_steps=args.num_ddim_steps, eta=0.0,
            device=device,
        )
        results.append(r)
        status = "✓" if r["success"] else "✗"
        print(f"  {status}  len={r['length']:3d}  reward={r['reward']:.2f}")

    print_stats(results)
    plot_trajectories(results, args.out)


if __name__ == "__main__":
    main()
