def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)

    # Joint position command (deviation from default joint positions)
    self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
    self._previous_actions = torch.zeros(
        self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
    )
    
    # part 2
    # PD control params
    self.Kp = torch.tensor([cfg.Kp]*12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
    self.Kd = torch.tensor([cfg.Kd]*12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
    self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
    self.torque_limits = cfg.torque_limits
    
    # X/Y linear velocity and yaw angular velocity commands
    self._commands = torch.zeros(self.num_envs, 3, device=self.device)

    # Logging
    self._episode_sums = {
        key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for key in [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "rew_action_rate",
            "raibert_heuristic",
            "orient",       # STABILITY
            "lin_vel_z",    # STABILITY
            "ang_vel_xy",   # STABILITY
            "dof_vel",      # ACTION REG & SMOOTH
            "torque",       # ACTION REG & SMOOTH
        ]
    }
    # Get specific body indices
    self._base_id, _ = self._contact_sensor.find_bodies("base")

    # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
    self.set_debug_vis(self.cfg.debug_vis)
    
    # part 1.2
    self.last_actions = torch.zeros(
        self.num_envs, 
        gym.spaces.flatdim(self.single_action_space),
        3, 
        dtype=torch.float, 
        device=self.device, 
        requires_grad=False
    )
                
    # part 4 
    # Get specific body indices
    self._feet_ids = []
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    for name in foot_names:
        id_list, _ = self.robot.find_bodies(name)
        self._feet_ids.append(id_list[0])
        
    # variables needed for the raibert heuristic
    self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
    self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)