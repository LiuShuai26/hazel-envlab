"""
Hazel EnvLab - Training Example with Stable Baselines 3

This script demonstrates how to train RL agents using the Hazel game engine.

## Parallel Training Fields

EnvLab supports multiple parallel training fields in a single scene for
faster sample collection. Each field is spatially separated, and when one
field's episode ends, only that field resets.

NOTE: This is NOT multi-agent RL (agents don't interact with each other).
It's parallel data collection - multiple copies of the same environment
training the same policy simultaneously.

```
Single Scene Layout:
┌────────────────────────────────────────────────────────────┐
│  Field 0 (x=0)      Field 1 (x=100)     Field 2 (x=200)   │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     │
│  │ Agent0     │     │ Agent1     │     │ Agent2     │     │
│  │ Obstacles  │     │ Obstacles  │     │ Obstacles  │     │
│  │ Ground     │     │ Ground     │     │ Ground     │     │
│  └────────────┘     └────────────┘     └────────────┘     │
└────────────────────────────────────────────────────────────┘
```

## Scene Setup

Each training field must use parent-child hierarchy:

    TrainingField0 (empty parent entity)
    ├── Agent0 (has AgentComponent + ScriptComponent)
    ├── Ground0 (static collider)
    ├── Obstacle_A
    └── Obstacle_B

    TrainingField1 (empty parent entity)
    ├── Agent1 (has AgentComponent + ScriptComponent)
    ├── Ground1 (static collider)
    └── ...

When Field0's episode ends (Agent0 Done=true), only Field0 resets while
Field1 continues uninterrupted. N fields = N samples per step.

## Requirements

    pip install stable-baselines3 gymnasium

## Usage

    1. Open Hazel editor
    2. Load your scene with training fields (see setup above)
    3. Start Step Runtime with Training enabled
    4. Run: python train_example.py [--vectorized]
"""

import argparse
from hazel_envlab import HazelEnv, HazelVecEnv, HazelDisconnectedError

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Stable Baselines 3 not installed. Install with: pip install stable-baselines3")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def convert_to_onnx(model, output_path: str, obs_size: int):
    """
    Convert SB3 PPO model to ONNX format.

    Args:
        model: Trained SB3 PPO model
        output_path: Path for the output .onnx file
        obs_size: Observation space size
    """
    if not HAS_TORCH:
        print("PyTorch not installed. Skipping ONNX export.")
        return False

    # Define OnnxablePolicy here so it's only created when torch is available
    class OnnxablePolicy(nn.Module):
        """Wrapper to make SB3 policy exportable to ONNX."""

        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, observation):
            features = self.policy.extract_features(observation)
            if hasattr(self.policy, 'mlp_extractor'):
                latent_pi, _ = self.policy.mlp_extractor(features)
            else:
                latent_pi = features
            action_logits = self.policy.action_net(latent_pi)
            return torch.argmax(action_logits, dim=1)

    print(f"\nExporting model to ONNX: {output_path}")
    print(f"Observation size: {obs_size}")

    # Create exportable policy
    policy = model.policy.to("cpu")
    policy.eval()

    onnx_policy = OnnxablePolicy(policy)
    onnx_policy.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_size)

    # Export to ONNX
    # Use dynamo=False for legacy TorchScript exporter (more compatible with older runtimes)
    # Use opset_version=18 to avoid version conversion warnings
    try:
        torch.onnx.export(
            onnx_policy,
            dummy_input,
            output_path,
            opset_version=18,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"}
            },
            dynamo=False,  # Use legacy TorchScript exporter for compatibility
        )
    except TypeError:
        # Older PyTorch versions don't have dynamo parameter
        torch.onnx.export(
            onnx_policy,
            dummy_input,
            output_path,
            opset_version=18,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"}
            },
        )

    print(f"ONNX model saved to: {output_path}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except ImportError:
        print("Note: Install 'onnx' package for model verification")
    except Exception as e:
        print(f"ONNX verification warning: {e}")

    return True


class TrainingMetricsCallback(BaseCallback):
    """Callback for logging per-agent training metrics."""

    def __init__(self, num_agents: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.num_agents = num_agents
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = [0.0] * num_agents
        self.current_lengths = [0] * num_agents

    def _on_step(self) -> bool:
        import numpy as np

        # Get rewards and dones for all agents
        rewards = self.locals.get("rewards", [0.0] * self.num_agents)
        dones = self.locals.get("dones", [False] * self.num_agents)

        # Convert numpy arrays to flat lists
        if isinstance(rewards, np.ndarray):
            rewards = rewards.flatten().tolist()
        elif not isinstance(rewards, (list, tuple)):
            rewards = [float(rewards)]

        if isinstance(dones, np.ndarray):
            dones = dones.flatten().tolist()
        elif not isinstance(dones, (list, tuple)):
            dones = [bool(dones)]

        # Accumulate per-agent rewards
        for i in range(min(len(rewards), self.num_agents)):
            self.current_rewards[i] += float(rewards[i])
            self.current_lengths[i] += 1

            if i < len(dones) and bool(dones[i]):
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL agents with Hazel EnvLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--host", default="localhost",
        help="RLServer host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="RLServer port (default: 5555)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=200000,
        help="Total training timesteps (default: 200000)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--vectorized", action="store_true",
        help="Use vectorized environment (treats each training field as parallel env)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)"
    )
    parser.add_argument(
        "--save-path", default="hazel_agent_ppo",
        help="Path to save trained model (default: hazel_agent_ppo)"
    )
    parser.add_argument(
        "--onnx-path", default="hazel_agent.onnx",
        help="Path to save ONNX model (default: hazel_agent.onnx)"
    )
    parser.add_argument(
        "--no-onnx", action="store_true",
        help="Skip ONNX export after training"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Hazel EnvLab - RL Training with Parallel Training Fields")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Open Hazel editor")
    print("2. Load scene with training fields (parent-child hierarchy)")
    print("3. Start Step Runtime with 'Training' checkbox enabled")
    print("4. This script will connect and start training")
    print()
    print("Training Field Setup (for parallel training):")
    print("  TrainingField0 (parent)")
    print("  ├── Agent0 (AgentComponent + ScriptComponent)")
    print("  ├── Ground, Obstacles, etc.")
    print("  TrainingField1 (parent, spatially separated)")
    print("  ├── Agent1 ...")
    print()

    # Create environment
    print(f"Connecting to Hazel at {args.host}:{args.port}...")

    try:
        if args.vectorized:
            # Vectorized environment - treats each training field as a parallel sub-env
            env = HazelVecEnv(host=args.host, port=args.port)
            num_agents = env.num_envs
            print(f"Using vectorized environment with {num_agents} parallel training fields")
        else:
            # Standard environment (uses first agent only)
            env = HazelEnv(host=args.host, port=args.port)
            num_agents = len(env.agent_ids)
            print(f"Using single-field environment (scene has {num_agents} training fields)")

            if not env.connected:
                print("ERROR: Could not connect to Hazel RLServer!")
                print("Make sure Hazel is running in Step Runtime with Training enabled.")
                return
    except (ConnectionError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    if not HAS_SB3:
        print("Stable Baselines 3 not available. Running manual test loop...")
        manual_test(env)
        return

    # Validate environment (only for non-vectorized)
    if not args.vectorized:
        print("Validating environment...")
        try:
            check_env(env, warn=True)
            print("Environment validation passed!")
        except Exception as e:
            print(f"Environment validation warning: {e}")

    # Create PPO model
    print(f"\nCreating PPO model (lr={args.learning_rate})...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print(f"\nTraining for {args.timesteps} timesteps...")
    print("Press Ctrl+C to stop training\n")

    # Train
    obs_size = env.observation_space.shape[0]
    training_complete = False

    try:
        callback = TrainingMetricsCallback(num_agents=num_agents)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True
        )
        print("\nTraining complete!")
        training_complete = True

        # Save model
        model.save(args.save_path)
        print(f"Model saved to {args.save_path}.zip")

        # Export to ONNX
        if not args.no_onnx:
            convert_to_onnx(model, args.onnx_path, obs_size)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C)")
        # Save partial model
        model.save(args.save_path + "_partial")
        print(f"Partial model saved to {args.save_path}_partial.zip")

        # Still export ONNX for partial model if requested
        if not args.no_onnx:
            partial_onnx = args.onnx_path.replace(".onnx", "_partial.onnx")
            convert_to_onnx(model, partial_onnx, obs_size)

    except HazelDisconnectedError as e:
        print(f"\nLost connection to Hazel: {e}")
        print("Saving partial model...")
        model.save(args.save_path + "_partial")
        print(f"Partial model saved to {args.save_path}_partial.zip")

    finally:
        # Always close the environment
        env.close()

    # Evaluate (only if training completed and not vectorized)
    if training_complete and not args.vectorized:
        print(f"\nEvaluating trained agent ({args.eval_episodes} episodes)...")
        # Need to reconnect for evaluation
        eval_env = HazelEnv(host=args.host, port=args.port)
        if eval_env.connected:
            evaluate(eval_env, model, n_episodes=args.eval_episodes)
            eval_env.close()
        else:
            print("Could not reconnect for evaluation.")


def manual_test(env, n_steps: int = 100):
    """Manual test loop without SB3."""
    print(f"\nRunning {n_steps} random steps...")
    print("Press Ctrl+C to stop.\n")

    try:
        obs, info = env.reset()
        total_reward = 0
        episode = 1

        for step in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 20 == 0:
                obs_preview = obs[:3] if hasattr(obs, '__len__') and len(obs) > 3 else obs
                print(f"Step {step}: reward={reward:.4f}, obs={obs_preview}...")

            if terminated or truncated:
                print(f"Episode {episode} ended with total reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                episode += 1

        print(f"\nCompleted {n_steps} steps across {episode} episodes")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except HazelDisconnectedError as e:
        print(f"\nLost connection to Hazel: {e}")
    finally:
        env.close()


def evaluate(env, model, n_episodes: int = 5):
    """Evaluate a trained model."""
    episode_rewards = []
    episode_lengths = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Eval episode {ep + 1}: reward = {total_reward:.2f}, length = {steps}")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user (Ctrl+C)")
    except HazelDisconnectedError as e:
        print(f"\nLost connection during evaluation: {e}")

    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_length = sum(episode_lengths) / len(episode_lengths)
        print(f"\nEvaluation Results ({len(episode_rewards)} episodes):")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average length: {avg_length:.1f}")


if __name__ == "__main__":
    main()
