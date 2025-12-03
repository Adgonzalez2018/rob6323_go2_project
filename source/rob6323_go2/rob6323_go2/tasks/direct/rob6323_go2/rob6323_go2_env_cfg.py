# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.actuators import ImplicitActuatorCfg  # Fixed import
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # Added 4 for clock inputs (moved from later)
    state_space = 0
    debug_vis = True
    
    # STABILITY
    # reward scales: 
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    ang_vel_xy_reward_scale = -0.001
    
    # ACTION REGULARIZATION AND SMOOTHNESS
    dof_vel_reward_scale = -0.0001   # small val, penalize high joint velocities
    torque_reward_scale = -0.00001   # smaller val, penalize high torques
    
    # BONUS PART 1
    # Actuator friction params
    friction_enabled = True	# set to False to disable friction
    mu_v_range = (0.0, 0.3)	# Viscous friction range
    F_s_range = (0.0,2.5)	# Stiction friction range
    
    
    # part 1
    # reward scales
    action_rate_reward_scale = -0.1
    
    # part 2
    # PD control gains
    Kp = 20.0                # proportional gain
    Kd = 0.5                 # derivative gain
    torque_limits = 100.0    # Max torque

    # part 3 
    base_height_min = 0.2    # terminate if base is lower than 20cm
    
    # part 4
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_reward_scale = -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0
    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # update robot_cfg - Part 2: Custom PD Controller
    # "base_legs" is an arbitrary key we use to group these actuators
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,   # critical: set to 0 to disable implicit p-gain
        damping=0.0,     # Critical: set to 0 to disable implicit d-gain
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        history_length=3, 
        update_period=0.005, 
        track_air_time=True
    )
    
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5