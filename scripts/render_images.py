#!/usr/bin/env python3
"""
Render RGB images from a Robomimic raw HDF5 dataset.

Reads env_args from the raw HDF5, recreates the robosuite environment
with offscreen rendering enabled, replays stored states, and saves
agentview images + proprioception to a new HDF5 file.

Usage:
    python scripts/render_images.py \
        --input_path  data/robomimic/lift_ph_raw.hdf5 \
        --output_path data/robomimic/lift_ph_image_v15.hdf5 \
        --img_size 84
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Headless rendering — must be set before any mujoco/GL import
os.environ.setdefault("MUJOCO_GL",        "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM","osmesa")

import h5py
import numpy as np
from tqdm import tqdm
import robosuite as suite
from robosuite import load_composite_controller_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",    type=str, required=True)
    p.add_argument("--output_path",   type=str, required=True)
    p.add_argument("--img_size",      type=int, default=84)
    p.add_argument("--render_camera", type=str, default="agentview")
    p.add_argument("--max_demos",     type=int, default=None)
    return p.parse_args()


def build_env(env_args_str: str, img_size: int, camera: str):
    """
    Parse env_args from the HDF5 and create a robosuite environment
    with offscreen rendering and camera observations enabled.
    """
    env_args = json.loads(env_args_str)
    kwargs   = env_args.get("env_kwargs", {})
    env_name = env_args.get("env_name", "Lift")

    # Enable rendering (raw dataset had these disabled)
    kwargs["has_renderer"]           = False
    kwargs["has_offscreen_renderer"] = True
    kwargs["use_camera_obs"]         = True
    kwargs["camera_names"]           = [camera, "robot0_eye_in_hand"]
    kwargs["camera_heights"]         = img_size
    kwargs["camera_widths"]          = img_size
    kwargs["camera_depths"]          = False
    kwargs["ignore_done"]            = True
    kwargs["reward_shaping"]         = True

    # controller_configs key changed in newer robosuite — use as-is
    env = suite.make(env_name, **kwargs)
    return env


def render_demo(env, states: np.ndarray, camera: str):
    """Replay MuJoCo states and collect images + proprioception."""
    images, eef_pos, eef_quat, gripper = [], [], [], []

    env.reset()
    for t in range(len(states)):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        obs = env._get_observations(force_update=True)

        images.append(obs[f"{camera}_image"])          # (H,W,3) uint8

        eef_pos.append(obs.get("robot0_eef_pos",   np.zeros(3)))
        eef_quat.append(obs.get("robot0_eef_quat",  np.zeros(4)))
        gripper.append(obs.get("robot0_gripper_qpos", np.zeros(2)))

    return {
        f"{camera}_image":       np.array(images,   dtype=np.uint8),
        "robot0_eef_pos":        np.array(eef_pos,  dtype=np.float32),
        "robot0_eef_quat":       np.array(eef_quat, dtype=np.float32),
        "robot0_gripper_qpos":   np.array(gripper,  dtype=np.float32),
    }


def main():
    args = parse_args()
    src_path = Path(args.input_path)
    dst_path = Path(args.output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        print(f"Error: {src_path} not found."); sys.exit(1)

    print(f"\n{'═'*60}")
    print(f" Input:   {src_path.name}")
    print(f" Output:  {dst_path.name}")
    print(f" Camera:  {args.render_camera}  {args.img_size}px")
    print(f"{'═'*60}\n")

    # ── Load raw dataset ────────────────────────────────────────────────
    with h5py.File(src_path, "r") as src:
        env_args_str = src["data"].attrs["env_args"]
        demo_keys    = sorted(src["data"].keys())
        if args.max_demos:
            demo_keys = demo_keys[:args.max_demos]

        demos = {}
        for dk in tqdm(demo_keys, desc="Reading raw demos"):
            demos[dk] = {
                "states":  src["data"][dk]["states"][:],
                "actions": src["data"][dk]["actions"][:],
                "attrs":   dict(src["data"][dk].attrs),
            }

    # ── Build robosuite env (once, reused for all demos) ────────────────
    print("Building robosuite env …")
    env = build_env(env_args_str, args.img_size, args.render_camera)
    print("Ready.\n")

    # ── Render & write output HDF5 ──────────────────────────────────────
    with h5py.File(dst_path, "w") as dst:
        data_grp = dst.create_group("data")

        for dk in tqdm(demo_keys, desc="Rendering"):
            states  = demos[dk]["states"]
            actions = demos[dk]["actions"]

            obs_dict = render_demo(env, states, args.render_camera)

            grp = data_grp.create_group(dk)
            for k, v in demos[dk]["attrs"].items():
                grp.attrs[k] = v

            grp.create_dataset("actions", data=actions)

            # Compute rewards on-the-fly (replay with step)
            rewards = []
            env.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            for t in range(len(actions)):
                _, r, _, _ = env.step(actions[t])
                rewards.append(r)
            grp.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))
            grp.create_dataset("states",  data=states)

            obs_grp = grp.create_group("obs")
            for obs_key, obs_val in obs_dict.items():
                kw = {"compression": "gzip", "compression_opts": 4} \
                     if obs_key.endswith("_image") else {}
                obs_grp.create_dataset(obs_key, data=obs_val, **kw)

        # Train / valid split mask
        n_val   = max(1, int(len(demo_keys) * 0.1))
        train_k = demo_keys[:len(demo_keys) - n_val]
        valid_k = demo_keys[len(demo_keys) - n_val:]
        mask = dst.create_group("mask")
        mask.create_dataset("train", data=np.array(train_k, dtype=h5py.string_dtype()))
        mask.create_dataset("valid", data=np.array(valid_k, dtype=h5py.string_dtype()))

    env.close()
    size_mb = dst_path.stat().st_size / 1024**2
    print(f"\n✓ Done — {len(demo_keys)} demos → {dst_path}  ({size_mb:.1f} MB)")
    print(f"\nNext:\n  python scripts/train_visual.py --data {dst_path}")


if __name__ == "__main__":
    main()
