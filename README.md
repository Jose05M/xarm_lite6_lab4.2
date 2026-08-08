# xarm_lite6_lab4.2

ROS 2 package (Python, `ament_python`) developed for **Lab 4.2** of the *TE3001B – Fundamentación de Robótica* course (Tecnológico de Monterrey, Campus Monterrey). It implements a **Cartesian Proportional-Derivative (PD) controller** for the end-effector of the **xArm Lite 6**, using **MoveIt Servo** within ROS 2, and allows injecting perturbations (sinusoidal and Gaussian) to evaluate the controller's performance under nominal and perturbed conditions.

**Team: RJ CREW**

| Student | ID |
|---|---|
| Jose Eduardo Sanchez Martinez | A01738476 |
| Josue Ureña Valencia | A01738940 |
| Rafael André Gamiz Salazar | A00838280 |
| César Arellano Arellano | A00839373 |

Instructor: Nezih Nieto Gutiérrez

## Package contents

```
xarm_lite6_lab4.2/
├── xarm_perturbations/
│   ├── circle_maker.py          # Main node: generates the circular trajectory and the Cartesian PD controller
│   ├── perturbation_injector.py # Optional node: injects sinusoidal or Gaussian perturbations
│   └── __init__.py
├── resource/xarm_perturbations
├── test/
├── package.xml
├── setup.py / setup.cfg
├── Evidencias/                  # Tracking plots and CSVs obtained during testing
└── Robotics Control Lab 4.2.pdf # Lab report
```

## How it works

The system is made up of three blocks, described in detail in `Robotics Control Lab 4.2.pdf`:

1. **Trajectory generator** — `circle_maker.py` generates, on every cycle, a circular reference `(x_d, y_d, z_d)` in the `link_base` frame, centered on the end-effector's initial pose.
2. **Cartesian PD controller** — on every cycle the current end-effector pose (`link_eef`) is read via TF, the error `error = target_pos - current` is computed, and a velocity command `v = Kp*error + Kd*d_error` is generated and published as `TwistStamped` on `/servo_server/delta_twist_cmds` (consumed by MoveIt Servo, which converts it into joint velocities via the manipulator's Jacobian).
3. **Perturbation injection** — `perturbation_injector.py` independently publishes, on the same `/servo_server/delta_twist_cmds` topic, an additional perturbation signal (`sine` or `gaussian`), allowing you to observe how `circle_maker.py`'s PD controller rejects/corrects the resulting error.

`circle_maker.py` keyboard controls (while the node's terminal has focus):
- `p` → pause / resume the circular motion.
- `h` → send the arm to the *home* position defined in `home_position`, then resume the circle.

`circle_maker.py` also logs to `tracking_data.csv` (created in the directory the node is launched from) the history of `time, x_d, y_d, z_d, x, y, z, ex, ey, ez` for later analysis, and computes the per-axis tracking RMSE.

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- [`xarm_ros2`](https://github.com/xArm-Developer/xarm_ros2) (UFACTORY), with xArm Lite 6 support and the `xarm_moveit_servo` package
- [`pymoveit2`](https://github.com/AndrejOrsula/pymoveit2)
- Python dependencies: `numpy`, `pynput` (`sudo apt install python3-pynput` or via `rosdep`), `tf2_ros` (ships with ROS 2)

## Workspace installation / setup

Follow the workspace installation/setup guide (cloning `xarm_ros2`, `rosdep`, `colcon build`, etc.) described in [lite6_demo_moveit/README.md](https://github.com/Jose05M/lite6_demo_moveit/blob/main/README.md).

Once the workspace is set up, place (or clone) this `xarm_lite6_lab4.2` package inside `~/xarm_ws/src/` and build it:

```bash
cd ~/xarm_ws/
colcon build --packages-select xarm_perturbations
source install/setup.bash
```

## How to launch it

First move the arm to a safe starting pose, then bring up MoveIt Servo for the xArm Lite 6 (real robot or simulation), and once it's running, launch this package's nodes in other terminal(s).

1. **Move the robot to a starting pose.** Before starting MoveIt Servo, the arm should be at a known, safe joint configuration — otherwise the circle gets centered on whatever pose the robot happens to be in, which may be too close to a singularity, a joint limit, or the edge of the workspace. Bring up regular MoveIt (not Servo) for this:

   ```bash
   ros2 launch xarm_moveit_config lite6_moveit_realmove.launch.py robot_ip:=192.168.1.179
   ```

   Adjust `robot_ip` to the arm controller's actual IP.

   **If the physical robot isn't available**, use the simulation/fake-hardware equivalent instead (no `robot_ip` needed):

   ```bash
   ros2 launch xarm_moveit_config lite6_moveit_fake.launch.py
   ```

   In RViz, under the MotionPlanning panel's **"Group joints of start state"** section, set each joint to a suitable starting value — for example the pose used for these tests (`joint1=0°, joint2=9°, joint3=44°, joint4=0°, joint5=35°, joint6=0°`) — then hit **Plan & Execute** to move the real (or simulated) robot there. Once it reaches that pose, close this MoveIt session before continuing to step 2.

2. **Launch MoveIt Servo.** With the real robot (make sure the controller is powered on and reachable on the network, and always keep the emergency stop button within reach):

   ```bash
   ros2 launch xarm_moveit_servo lite6_moveit_servo_realmove.launch.py robot_ip:=192.168.1.123
   ```

   Adjust `robot_ip` to the arm controller's actual IP.

   **If the physical robot isn't available**, use the simulation/fake-hardware equivalent instead (no `robot_ip` needed):

   ```bash
   ros2 launch xarm_moveit_servo lite6_moveit_servo_fake.launch.py
   ```

   Either option leaves MoveIt Servo's `servo_server` running, ready to receive `TwistStamped` commands on `/servo_server/delta_twist_cmds`.

3. **Run the main controller** (generates the circle and corrects the error via PD), in another terminal with the workspace already *sourced*:

   ```bash
    ros2 run xarm_perturbations circle_maker --ros-args   -p radius:=0.06   -p frequency:=0.06   -p plane:=xy   -p hold_z:=true
   ```

   The end-effector will start tracing the circle around its initial pose. Use `p`/`h` from that terminal to pause/go home.

4. **Inject a perturbation**, in another terminal, while `circle_maker` keeps running:

   ```bash
   ros2 run xarm_perturbations perturbation_injector --ros-args -p mode:=sine -p sine_freq_hz:=8.0 -p sine_amp_linear:=0.02
   ```

   Switch `mode` to `gaussian` (with `noise_std_linear`) for the stochastic perturbation, or to `off` to disable it without killing the node.

5. **(Optional) Watch the plots live**, in another terminal, with `rqt_plot` (same as the screenshots in `Robotics Control Lab 4.2.pdf`):

   ```bash
   ros2 run rqt_plot rqt_plot
   ```

   Or directly with the topics preloaded:

   ```bash
   ros2 run rqt_plot rqt_plot /servo_server/delta_twist_cmds/twist/linear/x /servo_server/delta_twist_cmds/twist/linear/y /servo_server/delta_twist_cmds/twist/linear/z
   ```

   This plots, in real time, the `x`, `y`, `z` components of the Cartesian velocity command published by `circle_maker` (mixed with whatever `perturbation_injector` publishes, if it's running), letting you compare nominal vs. perturbed behavior.

## Evidence

The `Evidencias/` folder contains the plots (`Elipse sin perturbaciones.png`, `Elipse con seno.png`, `Elipse con gauss.png`) and the tracking CSVs (`tracking_data*.csv`) generated during the tests documented in `Robotics Control Lab 4.2.pdf`, which covers the full development of the trajectory generator, the PD controller, the perturbation injection, and the measurement error analysis.

## Known notes / limitations

- `circle_maker.py` and `perturbation_injector.py` publish to the **same topic** (`/servo_server/delta_twist_cmds`); the perturbation isn't explicitly summed into the PD command — both commands are interleaved, and it's the PD loop (with TF feedback) that corrects the resulting error.
- The output CSV path (`tracking_data.csv`) is relative to the directory `circle_maker` is run from; it isn't fixed to an absolute path.
- `home_position` in `circle_maker.py` is *hardcoded*; adjust it if your setup/working environment changes before using the `h` control.
