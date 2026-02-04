# hazel-envlab

Python client library for Hazel EnvLab - Train RL agents in the Hazel game engine with Gym-compatible environments.

## Features

- **Gym-Compatible Environments** - Standard `reset()`, `step()`, `observation_space`, `action_space` API
- **Parallel Training Fields** - Multiple spatially-separated environments in one scene for faster sample collection
- **Stable Baselines 3 Integration** - Ready-to-use with PPO and other SB3 algorithms
- **ONNX Export** - Convert trained models for inference in Hazel runtime
- **Discrete & Continuous Actions** - Support for discrete, continuous, and hybrid action spaces

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Setup Hazel Scene

Create a training field with parent-child hierarchy:

```
TrainingField (empty parent entity)
├── Agent (AgentComponent + ScriptComponent)
├── Ground (static collider)
├── Obstacle_A
└── Obstacle_B
```

### 2. Start Hazel in Step Runtime

1. Open Hazel editor
2. Load your scene
3. Open EnvLab panel (View → EnvLab)
4. Enable "Training Mode"
5. Click "Start Step Runtime"

### 3. Connect and Train

```python
from hazel_envlab import HazelEnv

# Connect to Hazel
env = HazelEnv(host="localhost", port=5555)

# Standard Gym loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Training with Stable Baselines 3

```python
from hazel_envlab import HazelEnv
from stable_baselines3 import PPO

env = HazelEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("my_agent")
```

Or use the included training script:

```bash
python train.py --timesteps 200000 --onnx-path my_agent.onnx
```

## Parallel Training Fields

For faster training, create multiple training fields in one scene:

```
Scene
├── TrainingField_0 (at x=0)
│   ├── Agent_0
│   └── ...
├── TrainingField_1 (at x=100)
│   ├── Agent_1
│   └── ...
└── TrainingField_2 (at x=200)
    ├── Agent_2
    └── ...
```

Use `HazelVecEnv` for parallel sample collection:

```python
from hazel_envlab import HazelVecEnv
from stable_baselines3 import PPO

env = HazelVecEnv()  # Treats each field as parallel sub-environment
print(f"Training with {env.num_envs} parallel environments")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
```

Or use the included training script:

```bash
python train.py --vectorized --timesteps 200000
```

## Environment Classes

### HazelEnv

Standard single-agent Gym environment. Uses the first agent if multiple exist.

```python
env = HazelEnv(host="localhost", port=5555)
```

### HazelVecEnv

Vectorized environment for parallel training fields. Compatible with SB3 `VecEnv`.

```python
env = HazelVecEnv(host="localhost", port=5555)
# env.num_envs = number of training fields
```

## ONNX Export

Export trained models for inference in Hazel:

```python
from train import convert_to_onnx
from stable_baselines3 import PPO

model = PPO.load("my_agent.zip")
convert_to_onnx(model, "my_agent.onnx", obs_size=5)
```

Or use the training script which exports automatically:

```bash
python train.py --onnx-path my_agent.onnx
```

## Deploy to Hazel

After training and exporting to ONNX, deploy the model back to Hazel for inference:

### 1. Copy ONNX Model to Project

Copy your exported `.onnx` file to your Hazel project's Assets folder:

```
MyProject/
├── Assets/
│   ├── Models/
│   │   └── my_agent.onnx    <-- Place here
│   └── ...
└── MyProject.hproj
```

### 2. Configure Agent for Inference

In the Hazel editor, select your Agent entity and set the AgentComponent properties:

| Property | Value |
|----------|-------|
| Behavior Mode | `Inference` |
| ONNX Model | Select your `.onnx` file |

### 3. Run in Play Mode

1. Switch from "Step Runtime" to normal "Play" mode
2. The agent will now use the trained ONNX model for decisions
3. No Python connection needed - inference runs entirely in Hazel

### Behavior Modes

| Mode | Description |
|------|-------------|
| **Training** | Actions from Python (Step Runtime + training script) |
| **Inference** | Actions from ONNX model (standalone, no Python) |
| **Heuristic** | Actions from `Heuristic()` method (manual testing) |

