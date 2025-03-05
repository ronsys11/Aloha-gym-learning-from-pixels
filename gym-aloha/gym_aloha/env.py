#env.py
import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces
import matplotlib.pyplot as plt
from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=240,
        observation_height=140,
        visualization_width=240,
        visualization_height=140,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        # Define action space based on task type with smaller bounds
        if self.task == "end_effector_transfer_cube":
            # Position bounds (xyz): smaller range for more stable control
            pos_low = np.array([-0.3, -0.3, -0.3])
            pos_high = np.array([0.3, 0.3, 0.3])
            
            # Quaternion bounds: normalized to [-1, 1]
            quat_low = np.array([-1.0, -1.0, -1.0, -1.0])
            quat_high = np.array([1.0, 1.0, 1.0, 1.0])
            
            # Gripper bounds: [0, 1] for close/open
            gripper_low = np.array([0.0])
            gripper_high = np.array([1.0])
            
            # Combine bounds for both arms
            low = np.concatenate([pos_low, quat_low, gripper_low] * 2)
            high = np.concatenate([pos_high, quat_high, gripper_high] * 2)
            
            self.action_space = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32
            )
        else:
            # Original action space for other tasks
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(ACTIONS),),
                dtype=np.float32
            )

        # Define observation space based on obs_type
        if self.obs_type == "top_only":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, observation_height, observation_width),
                    dtype=np.uint8,
                ),
            })
        elif self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "angle": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "vis": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                        }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

    def render(self):
        images = self._render(visualize=True)
        return np.concatenate([images["top"], images["angle"], images["vis"]], axis=1)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        images = {
        "top": self._env.physics.render(height=height, width=width, camera_id="top"),
        "angle": self._env.physics.render(height=height, width=width, camera_id="angle"),
        "vis": self._env.physics.render(height=height, width=width, camera_id="front_close"),
        }
        return images

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        """Format raw observations to match demonstration format."""
        if self.obs_type == "top_only":
            width, height = self.observation_width, self.observation_height
            top_view = self._env.physics.render(height=height, width=width, camera_id="top")
            # Ensure the format matches your demonstrations
            return {
                "pixels": top_view.transpose(2, 0, 1).astype(np.uint8)
            }
        elif self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            # Change this to match the observation space structure
            obs = {
                "top": raw_obs["images"]["top"].copy(),
                "angle": raw_obs["images"]["angle"].copy(),
                "vis": raw_obs["images"]["vis"].copy(),
            }
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {
                    "top": raw_obs["images"]["top"].copy(),
                    "angle": raw_obs["images"]["angle"].copy(),
                    "vis": raw_obs["images"]["vis"].copy(),
                },
                "agent_pos": raw_obs["qpos"],
            }
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")
        return obs


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # Handle all task types
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))
        elif self.task == "end_effector_transfer_cube":
            # End effector tasks handle their own initialization
            pass
        else:
            raise ValueError(f"Unknown task: {self.task}")

        raw_obs = self._env.reset()
        observation = self._format_raw_obs(raw_obs.observation)
        info = {"is_success": False}
        return observation, info

    def step(self, action):
    # Clip actions to prevent extreme movements
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Add stronger action smoothing
        if not hasattr(self, '_last_action'):
            self._last_action = action.copy()
        else:
            action = 0.95 * self._last_action + 0.05 * action  # More aggressive smoothing
        self._last_action = action.copy()
        
        # Add small random noise to prevent getting stuck
        action += np.random.normal(0, 0.01, size=action.shape)
        
        _, reward, _, raw_obs = self._env.step(action)
        terminated = is_success = reward == 4
        info = {"is_success": is_success}
        observation = self._format_raw_obs(raw_obs)
        truncated = False
        
        return observation, reward, terminated, truncated, info

    def visualize_env(env):
        obs, _ = env.reset()

        # Render images from all camera angles
        images = env._render_all_views()

        # Create a subplot to visualize all views
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for ax, (view, image) in zip(axes, images.items()):
            ax.imshow(image)
            ax.set_title(f"{view.capitalize()} View")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def _render_all_views(self):
        """Render and return all camera views as a dictionary of images."""
        width, height = self.observation_width, self.observation_height
        return {
            "top": self._env.physics.render(height=height, width=width, camera_id="top"),
            "angle": self._env.physics.render(height=height, width=width, camera_id="angle"),
            "vis": self._env.physics.render(height=height, width=width, camera_id="front_close"),
        }


    def close(self):
        pass
