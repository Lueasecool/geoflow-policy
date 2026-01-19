if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import pathlib
import torch
import hydra
from omegaconf import OmegaConf
from termcolor import cprint

from canonical_policy.workspace.train_canonical_workspace import TrainCanonicalWorkspace
from canonical_policy.env_runner.base_image_runner import BaseImageRunner


def main():
    parser = argparse.ArgumentParser(description="Load a trained checkpoint, run rollout, and save a high-quality video.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("--output_dir", type=str, default=, help="Directory to save media/videos. Defaults to the checkpoint's parent output dir")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--n_test", type=int, default=1, help="Number of test seeds to run")
    parser.add_argument("--n_test_vis", type=int, default=5, help="Number of test videos to record")
    parser.add_argument("--n_train", type=int, default=0, help="Number of train demos to run (0 to skip)")
    parser.add_argument("--n_train_vis", type=int, default=0, help="Number of train videos to record")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (robosuite base is 20)")
    parser.add_argument("--crf", type=int, default=18, help="Video quality CRF (lower is higher quality; typical 16-24)")
    parser.add_argument("--sync", action="store_true", help="Force sync vector env to avoid multiprocessing issues")
    parser.add_argument("--render_w", type=int, default=1080, help="Render width in pixels")
    parser.add_argument("--render_h", type=int, default=1080, help="Render height in pixels")
    parser.add_argument("--camera", type=str, default="agentview", help="Render camera name")
    args = parser.parse_args()

    ckpt_path = pathlib.Path(os.path.expanduser(args.ckpt)).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load workspace and weights from checkpoint (exclude optimizer state for eval)
    ws: TrainCanonicalWorkspace = TrainCanonicalWorkspace.create_from_checkpoint(
        str(ckpt_path), exclude_keys=("optimizer",))

    # Resolve output directory
    if args.output_dir is not None:
        ws._output_dir = str(pathlib.Path(os.path.expanduser(args.output_dir)).resolve())
    out_dir = pathlib.Path(ws.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch env runner cfg to control video saving granularity and quality
    cfg = ws.cfg
    # These fields depend on the specific runner in cfg; set if present
    er = cfg.task.env_runner
    if hasattr(er, "n_test"):
        er.n_test = args.n_test
    if hasattr(er, "n_test_vis"):
        er.n_test_vis = args.n_test_vis
    if hasattr(er, "n_train"):
        er.n_train = args.n_train
    if hasattr(er, "n_train_vis"):
        er.n_train_vis = args.n_train_vis
    # Build instantiate kwargs to avoid mutating struct cfg
    instantiate_kwargs = {
        'output_dir': ws.output_dir,
        'fps': args.fps,
        'crf': args.crf,
        'render_hw': (args.render_h, args.render_w),
        'render_camera_name': args.camera,
        'n_test': args.n_test,
        'n_test_vis': args.n_test_vis,
        'n_train': args.n_train,
        'n_train_vis': args.n_train_vis,
    }

    # Optional: force runner to use SyncVectorEnv to avoid multiprocessing issues
    if args.sync:
        try:
            from canonical_policy.env_runner import robomimic_image_runner as rim
            from canonical_policy.gym_util.sync_vector_env import SyncVectorEnv

            class PatchedSyncVectorEnv(SyncVectorEnv):
                def __init__(self, env_fns, dummy_env_fn=None, observation_space=None, action_space=None, copy=True, **kwargs):
                    super().__init__(env_fns, observation_space=observation_space, action_space=action_space, copy=copy)

            rim.AsyncVectorEnv = PatchedSyncVectorEnv
        except Exception as e:
            cprint(f"Failed to switch to SyncVectorEnv: {e}", "red")

    # Instantiate env runner from cfg
    env_runner: BaseImageRunner = hydra.utils.instantiate(er, **instantiate_kwargs)
    assert isinstance(env_runner, BaseImageRunner)

    # Select policy (EMA or raw)
    policy = ws.model
    if args.use_ema and (ws.ema_model is not None):
        policy = ws.ema_model

    # Device and eval mode
    device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    # Ensure normalizer is set if dataset provided it during training
    # (ws.load_payload already restored it from checkpoint.)

    # Run rollout
    cprint("Starting rollout and video recording...", "yellow")
    runner_log = env_runner.run(policy)

    # Print a short summary and point to media directory
    media_dir = out_dir.joinpath("media")
    cprint(f"Videos saved to: {str(media_dir)}", "green")
    # Also show key numeric results if any
    for key, value in runner_log.items():
        if isinstance(value, (float, int)):
            cprint(f"{key}: {value:.4f}", "cyan")


if __name__ == "__main__":
    main()


