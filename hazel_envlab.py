"""
Hazel EnvLab - Gym Environment Wrapper

This module provides a standard OpenAI Gym interface for the Hazel game engine's
EnvLab reinforcement learning system.

Usage:
    from hazel_envlab import HazelEnv, HazelDisconnectedError

    # Using context manager (recommended - handles cleanup on Ctrl+C)
    try:
        with HazelEnv(host="localhost", port=5555) as env:
            obs, info = env.reset()
            for _ in range(1000):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
    except KeyboardInterrupt:
        print("Training interrupted")
    except HazelDisconnectedError:
        print("Lost connection to Hazel")

    # Or manual cleanup
    env = HazelEnv()
    try:
        # ... training loop ...
    finally:
        env.close()
"""

import atexit
import socket
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class HazelDisconnectedError(ConnectionError):
    """Raised when connection to Hazel RLServer is lost."""
    pass


class HazelEnv(gym.Env):
    """
    OpenAI Gym environment wrapper for Hazel EnvLab.

    Connects to the Hazel editor's RLServer via TCP and provides a standard
    Gym interface for reinforcement learning.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 30.0,
        auto_connect: bool = True,
    ):
        """
        Initialize the Hazel environment.

        Args:
            host: RLServer host address
            port: RLServer port number
            timeout: Socket timeout in seconds
            auto_connect: Whether to connect automatically on init
        """
        super().__init__()

        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.connected = False

        # Agent info (populated after connect)
        self.agents: List[Dict] = []
        self.agent_ids: List[int] = []
        self.obs_size: int = 0
        self.action_branches: List[int] = []      # Discrete action branches
        self.continuous_action_size: int = 0       # Number of continuous actions

        # Spaces (set after getting agent info)
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None

        # Register cleanup on exit
        atexit.register(self._cleanup)

        if auto_connect:
            self.connect()

    def _cleanup(self):
        """Cleanup handler for atexit."""
        if self.connected:
            self.close()

    def connect(self) -> bool:
        """Connect to the Hazel RLServer."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True

            # Verify connection
            response = self._send_command("ping")
            if response != "pong":
                raise ConnectionError(f"Unexpected ping response: {response}")

            # Get agent info and set up spaces
            self._fetch_agent_info()
            self._setup_spaces()

            print(f"Connected to Hazel RLServer at {self.host}:{self.port}")
            print(f"Agents: {len(self.agents)}")
            print(f"Observation size: {self.obs_size}")
            print(f"Action branches: {self.action_branches}")

            return True

        except Exception as e:
            print(f"Failed to connect to Hazel RLServer: {e}")
            self.connected = False
            return False

    def _send_command(self, command: str) -> str:
        """Send a command to the server and receive response."""
        if not self.socket or not self.connected:
            raise HazelDisconnectedError("Not connected to server")

        try:
            # Send command with newline terminator
            self.socket.sendall((command + "\n").encode())

            # Receive response
            response = b""
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    # Server closed connection
                    self._mark_disconnected()
                    raise HazelDisconnectedError("Server closed connection")
                response += chunk
                if b"\n" in chunk:
                    break

            return response.decode().strip()

        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for response to '{command}'")
        except (socket.error, OSError) as e:
            self._mark_disconnected()
            raise HazelDisconnectedError(f"Connection lost: {e}")

    def _mark_disconnected(self):
        """Mark the environment as disconnected."""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

    def _fetch_agent_info(self):
        """Fetch agent information from the server."""
        response = self._send_command("info")
        data = json.loads(response)

        self.agents = data.get("agents", [])
        if self.agents:
            # Use first agent's config (assuming single-agent for now)
            agent = self.agents[0]
            self.agent_ids = [a["id"] for a in self.agents]
            self.obs_size = agent.get("obs_size", 0)
            self.action_branches = agent.get("action_branches", [])
            self.continuous_action_size = agent.get("continuous_action_size", 0)

    def _setup_spaces(self):
        """Set up observation and action spaces based on agent info."""
        # Observation space: continuous box
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        # Action space depends on action types available
        has_discrete = len(self.action_branches) > 0
        has_continuous = self.continuous_action_size > 0

        if has_discrete and has_continuous:
            # Hybrid action space (Dict with both discrete and continuous)
            discrete_space = (spaces.Discrete(self.action_branches[0])
                              if len(self.action_branches) == 1
                              else spaces.MultiDiscrete(self.action_branches))
            continuous_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.continuous_action_size,),
                dtype=np.float32
            )
            self.action_space = spaces.Dict({
                "discrete": discrete_space,
                "continuous": continuous_space
            })
        elif has_continuous:
            # Continuous only
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.continuous_action_size,),
                dtype=np.float32
            )
        elif has_discrete:
            # Discrete only
            if len(self.action_branches) == 1:
                self.action_space = spaces.Discrete(self.action_branches[0])
            else:
                self.action_space = spaces.MultiDiscrete(self.action_branches)
        else:
            # Fallback: single discrete action
            self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        if not self.connected:
            self.connect()

        response = self._send_command("reset")
        data = json.loads(response)

        obs = self._parse_observations(data.get("obs", {}))
        info = data.get("info", {})

        return obs, info

    def step(
        self,
        action: Union[int, List[int], np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (int for Discrete, list for MultiDiscrete)

        Returns:
            observation: New observation
            reward: Reward from this step
            terminated: Whether episode ended (e.g., goal reached or failure)
            truncated: Whether episode was cut short (e.g., max steps)
            info: Additional information
        """
        if not self.connected:
            raise ConnectionError("Not connected to server")

        # Format action as JSON
        action_dict = self._format_action(action)
        command = f"step {json.dumps(action_dict)}"

        response = self._send_command(command)

        if response.startswith("error:"):
            raise RuntimeError(f"Step error: {response}")

        data = json.loads(response)

        obs = self._parse_observations(data.get("obs", {}))
        reward = float(data.get("reward", 0.0))
        info = data.get("info", {})

        # Parse terminated and truncated from server response
        # terminated = episode ended due to agent action (e.g., collision, goal reached)
        # truncated = episode ended due to time limit (max steps)
        terminated = bool(data.get("terminated", False))
        truncated = bool(data.get("truncated", False))

        return obs, reward, terminated, truncated, info

    def _format_action(self, action: Union[int, List[int], np.ndarray, Dict]) -> Dict:
        """Format action for sending to server."""
        has_discrete = len(self.action_branches) > 0
        has_continuous = self.continuous_action_size > 0

        action_dict = {}

        if has_discrete and has_continuous:
            # Hybrid action space - expect a dict with "discrete" and "continuous" keys
            if isinstance(action, dict):
                discrete_action = action.get("discrete", [0] * len(self.action_branches))
                continuous_action = action.get("continuous", [0.0] * self.continuous_action_size)
            else:
                # Fallback: treat as discrete only
                discrete_action = action
                continuous_action = [0.0] * self.continuous_action_size

            # Convert to lists
            if isinstance(discrete_action, (int, np.integer)):
                discrete_list = [int(discrete_action)]
            elif isinstance(discrete_action, np.ndarray):
                discrete_list = [int(x) for x in discrete_action.tolist()]
            else:
                discrete_list = [int(x) for x in discrete_action]

            if isinstance(continuous_action, np.ndarray):
                continuous_list = continuous_action.tolist()
            else:
                continuous_list = list(continuous_action)

            for agent_id in self.agent_ids:
                action_dict[str(agent_id)] = {
                    "discrete": discrete_list,
                    "continuous": continuous_list
                }

        elif has_continuous:
            # Continuous only
            if isinstance(action, np.ndarray):
                continuous_list = action.tolist()
            elif isinstance(action, (list, tuple)):
                continuous_list = list(action)
            else:
                continuous_list = [float(action)]

            for agent_id in self.agent_ids:
                action_dict[str(agent_id)] = {
                    "continuous": continuous_list
                }

        else:
            # Discrete only (backwards compatible)
            if isinstance(action, (int, np.integer)):
                action_list = [int(action)]
            elif isinstance(action, np.ndarray):
                action_list = [int(x) for x in action.tolist()]
            else:
                action_list = [int(x) for x in action]

            for agent_id in self.agent_ids:
                action_dict[str(agent_id)] = action_list

        return action_dict

    def _parse_observations(self, obs_dict: Dict) -> np.ndarray:
        """Parse observations from server response."""
        if not obs_dict:
            return np.zeros(self.obs_size, dtype=np.float32)

        # Get first agent's observations (single-agent assumption)
        for agent_id in self.agent_ids:
            key = str(agent_id)
            if key in obs_dict:
                obs = np.array(obs_dict[key], dtype=np.float32)
                return obs

        return np.zeros(self.obs_size, dtype=np.float32)

    def render(self):
        """Render is handled by Hazel editor."""
        pass

    def close(self):
        """Close the connection to the server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        if self.connected:
            self.connected = False
            print("Disconnected from Hazel RLServer")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False


class HazelMultiAgentEnv(HazelEnv):
    """
    Multi-agent variant of HazelEnv.

    Returns observations and accepts actions as dictionaries keyed by agent ID.
    """

    def _setup_spaces(self):
        """Set up observation and action spaces for multi-agent."""
        # Per-agent spaces
        single_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        if len(self.action_branches) == 1:
            single_action_space = spaces.Discrete(self.action_branches[0])
        else:
            single_action_space = spaces.MultiDiscrete(self.action_branches)

        # Dict spaces for multi-agent
        self.observation_space = spaces.Dict({
            str(aid): single_obs_space for aid in self.agent_ids
        })
        self.action_space = spaces.Dict({
            str(aid): single_action_space for aid in self.agent_ids
        })

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset and return dict of observations."""
        if seed is not None:
            super().reset(seed=seed)

        if not self.connected:
            self.connect()

        response = self._send_command("reset")
        data = json.loads(response)

        obs_dict = {}
        for agent_id in self.agent_ids:
            key = str(agent_id)
            if key in data.get("obs", {}):
                obs_dict[key] = np.array(data["obs"][key], dtype=np.float32)
            else:
                obs_dict[key] = np.zeros(self.obs_size, dtype=np.float32)

        return obs_dict, data.get("info", {})

    def step(self, action_dict: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step with dict of actions, return dict of observations."""
        if not self.connected:
            raise ConnectionError("Not connected to server")

        command = f"step {json.dumps(action_dict)}"
        response = self._send_command(command)

        if response.startswith("error:"):
            raise RuntimeError(f"Step error: {response}")

        data = json.loads(response)

        obs_dict = {}
        for agent_id in self.agent_ids:
            key = str(agent_id)
            if key in data.get("obs", {}):
                obs_dict[key] = np.array(data["obs"][key], dtype=np.float32)
            else:
                obs_dict[key] = np.zeros(self.obs_size, dtype=np.float32)

        reward = float(data.get("reward", 0.0))
        terminated = bool(data.get("terminated", False))
        truncated = bool(data.get("truncated", False))
        info = data.get("info", {})

        return obs_dict, reward, terminated, truncated, info


try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
    HAS_SB3_VECENV = True
except ImportError:
    HAS_SB3_VECENV = False
    VecEnv = object  # Fallback for type hints


class HazelVecEnv(VecEnv if HAS_SB3_VECENV else object):
    """
    Vectorized environment wrapper for Hazel EnvLab parallel training fields.

    This wraps a Hazel scene with multiple training fields as a Stable Baselines 3
    VecEnv, treating each training field as a parallel sub-environment.

    With N training fields in the scene, this acts as N parallel environments,
    enabling faster sample collection (N samples per step).

    Note: This is NOT multi-agent RL. Training fields are spatially separated
    and don't interact. All fields train the same policy in parallel.

    Key feature: Per-field episode resets. When one field's episode ends, only
    that field resets while other fields continue uninterrupted.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 30.0,
    ):
        if not HAS_SB3_VECENV:
            raise ImportError(
                "HazelVecEnv requires stable-baselines3. "
                "Install with: pip install stable-baselines3"
            )

        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.connected = False

        # Agent info
        self.agents: List[Dict] = []
        self.agent_ids: List[int] = []
        self.obs_size: int = 0
        self.action_branches: List[int] = []
        self.continuous_action_size: int = 0

        # Register cleanup on exit
        atexit.register(self._cleanup)

        # Connect and get info
        if not self.connect():
            raise ConnectionError(
                f"Failed to connect to Hazel RLServer at {host}:{port}. "
                "Make sure Hazel is running in Step Runtime with Training enabled."
            )

        # Initialize VecEnv base
        num_envs = len(self.agent_ids)
        if num_envs == 0:
            raise RuntimeError("No agents found in Hazel scene. Add at least one training field with an agent.")
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_size,), dtype=np.float32
        )

        # Action space depends on action types
        has_discrete = len(self.action_branches) > 0
        has_continuous = self.continuous_action_size > 0

        if has_discrete and has_continuous:
            # Hybrid action space - SB3 doesn't natively support Dict action spaces
            # Use flattened representation: [discrete..., continuous...]
            # For now, raise error - hybrid requires custom handling
            raise NotImplementedError(
                "HazelVecEnv does not yet support hybrid (discrete+continuous) action spaces. "
                "Use HazelEnv for hybrid action spaces or use only discrete or only continuous actions."
            )
        elif has_continuous:
            # Continuous only
            action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.continuous_action_size,),
                dtype=np.float32
            )
        elif has_discrete:
            if len(self.action_branches) == 1:
                action_space = spaces.Discrete(self.action_branches[0])
            else:
                action_space = spaces.MultiDiscrete(self.action_branches)
        else:
            # Fallback
            action_space = spaces.Discrete(2)

        # Set render_mode before super().__init__ to avoid SB3 warning
        self.render_mode = None

        super().__init__(num_envs, observation_space, action_space)

        # Per-agent tracking for auto-reset
        self._last_obs = np.zeros((num_envs, self.obs_size), dtype=np.float32)

    def connect(self) -> bool:
        """Connect to the Hazel RLServer."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True

            # Verify connection
            response = self._send_command("ping")
            if response != "pong":
                raise ConnectionError(f"Unexpected ping response: {response}")

            # Get agent info
            self._fetch_agent_info()

            print(f"HazelVecEnv: Connected to {self.host}:{self.port}")
            print(f"  Training fields (parallel envs): {len(self.agents)}")
            print(f"  Observation size: {self.obs_size}")
            print(f"  Discrete action branches: {self.action_branches}")
            print(f"  Continuous action size: {self.continuous_action_size}")

            return True

        except Exception as e:
            print(f"HazelVecEnv: Failed to connect: {e}")
            self.connected = False
            return False

    def _cleanup(self):
        """Cleanup handler for atexit."""
        if self.connected:
            self.close()

    def _mark_disconnected(self):
        """Mark the environment as disconnected."""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

    def _send_command(self, command: str) -> str:
        """Send a command to the server and receive response."""
        if not self.socket or not self.connected:
            raise HazelDisconnectedError("Not connected to server")

        try:
            self.socket.sendall((command + "\n").encode())

            response = b""
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    self._mark_disconnected()
                    raise HazelDisconnectedError("Server closed connection")
                response += chunk
                if b"\n" in chunk:
                    break

            return response.decode().strip()

        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for response to '{command}'")
        except (socket.error, OSError) as e:
            self._mark_disconnected()
            raise HazelDisconnectedError(f"Connection lost: {e}")

    def _fetch_agent_info(self):
        """Fetch agent information from the server."""
        response = self._send_command("info")
        data = json.loads(response)

        self.agents = data.get("agents", [])
        if self.agents:
            agent = self.agents[0]
            self.agent_ids = [a["id"] for a in self.agents]
            self.obs_size = agent.get("obs_size", 0)
            self.action_branches = agent.get("action_branches", [])
            self.continuous_action_size = agent.get("continuous_action_size", 0)

    def reset(self) -> VecEnvObs:
        """Reset all environments and return initial observations."""
        response = self._send_command("reset")
        data = json.loads(response)

        obs = np.zeros((self.num_envs, self.obs_size), dtype=np.float32)
        for i, agent_id in enumerate(self.agent_ids):
            key = str(agent_id)
            if key in data.get("obs", {}):
                obs[i] = np.array(data["obs"][key], dtype=np.float32)

        self._last_obs = obs.copy()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Start stepping the environments asynchronously."""
        self._pending_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """Wait for step to complete and return results."""
        actions = self._pending_actions

        # Determine action type
        has_discrete = len(self.action_branches) > 0
        has_continuous = self.continuous_action_size > 0

        # Format actions for each agent
        action_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            if i < len(actions):
                action = actions[i]

                if has_continuous and not has_discrete:
                    # Continuous only
                    if isinstance(action, np.ndarray):
                        continuous_list = action.tolist()
                    elif isinstance(action, (list, tuple)):
                        continuous_list = list(action)
                    else:
                        continuous_list = [float(action)]
                    action_dict[str(agent_id)] = {"continuous": continuous_list}
                else:
                    # Discrete only (hybrid is rejected in constructor)
                    if isinstance(action, np.ndarray):
                        action_list = [int(x) for x in action.tolist()]
                    elif isinstance(action, (int, np.integer)):
                        action_list = [int(action)]
                    else:
                        action_list = [int(x) for x in action]
                    action_dict[str(agent_id)] = action_list

        command = f"step {json.dumps(action_dict)}"
        response = self._send_command(command)

        if response.startswith("error:"):
            raise RuntimeError(f"Step error: {response}")

        data = json.loads(response)

        # Parse per-agent observations
        obs = np.zeros((self.num_envs, self.obs_size), dtype=np.float32)
        for i, agent_id in enumerate(self.agent_ids):
            key = str(agent_id)
            if key in data.get("obs", {}):
                obs[i] = np.array(data["obs"][key], dtype=np.float32)

        # Parse per-agent rewards (from per_agent_rewards if available, else split total)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        if "per_agent_rewards" in data:
            for i, agent_id in enumerate(self.agent_ids):
                key = str(agent_id)
                if key in data["per_agent_rewards"]:
                    rewards[i] = float(data["per_agent_rewards"][key])
        else:
            # Fallback: split total reward equally (not ideal)
            total_reward = float(data.get("reward", 0.0))
            rewards[:] = total_reward / max(1, self.num_envs)

        # Parse per-agent done states
        dones = np.zeros(self.num_envs, dtype=bool)
        if "per_agent_done" in data:
            for i, agent_id in enumerate(self.agent_ids):
                key = str(agent_id)
                if key in data["per_agent_done"]:
                    dones[i] = bool(data["per_agent_done"][key])
        else:
            # Fallback: use global terminated/truncated
            any_done = data.get("terminated", False) or data.get("truncated", False)
            dones[:] = any_done

        # Per-agent infos
        infos = [{} for _ in range(self.num_envs)]
        for i, done in enumerate(dones):
            if done:
                # Store terminal observation for SB3
                infos[i]["terminal_observation"] = self._last_obs[i].copy()

        self._last_obs = obs.copy()
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Close the connection."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        if self.connected:
            self.connected = False
            print("HazelVecEnv: Disconnected from Hazel RLServer")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set random seed (not implemented on server side)."""
        return [seed] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        """Check if environments are wrapped."""
        return [False] * self.num_envs

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call method on environments (not supported)."""
        raise NotImplementedError("env_method not supported for HazelVecEnv")

    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from environments (limited support)."""
        if attr_name == "observation_space":
            return [self.observation_space] * self.num_envs
        if attr_name == "action_space":
            return [self.action_space] * self.num_envs
        raise AttributeError(f"Attribute {attr_name} not available")

    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute on environments (not supported)."""
        raise NotImplementedError("set_attr not supported for HazelVecEnv")


# Convenience function to create environment
def make_hazel_env(
    host: str = "localhost",
    port: int = 5555,
    multi_agent: bool = False,
    vectorized: bool = False
) -> Union[HazelEnv, "HazelVecEnv"]:
    """
    Create a Hazel environment.

    Args:
        host: RLServer host
        port: RLServer port
        multi_agent: Whether to use multi-agent dict variant
        vectorized: Whether to use SB3-compatible vectorized environment
                    (treats each agent as parallel sub-env)

    Returns:
        HazelEnv, HazelMultiAgentEnv, or HazelVecEnv instance
    """
    if vectorized:
        return HazelVecEnv(host=host, port=port)
    if multi_agent:
        return HazelMultiAgentEnv(host=host, port=port)
    return HazelEnv(host=host, port=port)


# Test script
if __name__ == "__main__":
    print("Hazel EnvLab - Gym Environment Test")
    print("=" * 50)
    print("Make sure Hazel editor is running in Step Runtime with Training enabled!")
    print("Press Ctrl+C to stop.\n")

    env = None
    try:
        # Create environment using context manager for automatic cleanup
        with HazelEnv() as env:
            if not env.connected:
                print("Failed to connect. Is Hazel running in Step Runtime?")
                exit(1)

            # Test reset
            print("\nTesting reset...")
            obs, info = env.reset()
            print(f"Initial observation: {obs}")
            print(f"Info: {info}")

            # Test steps
            print("\nTesting steps...")
            total_reward = 0
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step {i+1}: action={action}, reward={reward:.4f}, obs={obs[:3]}...")

                if terminated or truncated:
                    print("Episode ended!")
                    obs, info = env.reset()
                    total_reward = 0

            print(f"\nTotal reward: {total_reward:.4f}")
            print("\nTest complete!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except HazelDisconnectedError as e:
        print(f"\nDisconnected from server: {e}")
    except Exception as e:
        print(f"\nError: {e}")
