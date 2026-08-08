


#!/usr/bin/env python3
import math
import numpy as np
import csv

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener
from pynput import keyboard
from enum import Enum


class RobotState(Enum):
    RUNNING = 1
    PAUSED = 2
    HOME = 3


class CircleServoXArmLite6(Node):
    """
    MoveIt Servo Cartesian controller that makes the xArm Lite 6 end-effector
    track a circle in link_base.
    """

    def __init__(self):
        super().__init__("circle_servo_xarm_lite6")
        self.csv_file = open("tracking_data_gauss.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "time", "x_d", "y_d", "z_d",
            "x", "y", "z", "ex", "ey", "ez"
        ])

        # ---------------------------
        # State machine
        # ---------------------------
        self.robot_state = RobotState.RUNNING

        # ---------------------------
        # Publisher to MoveIt Servo
        # ---------------------------
        self.servo_pub = self.create_publisher(
            TwistStamped, "/servo_server/delta_twist_cmds", 10
        )

        # ---------------------------
        # TF for current EE pose
        # ---------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------
        # xArm Lite 6-friendly defaults
        # ---------------------------
        self.radius = 0.20          # 4 cm circle
        self.frequency = 0.1       # 0.10 Hz -> ~10s per loop
        self.plane = "xy"           # draw in XY plane
        self.hold_z = True          # keep z constant (use center.z)
        self.rmse_sum = np.zeros(3)
        self.rmse_count = 0
        self.rmse_start_delay = 3.0

        # PD gains (task-space)
        self.kp = np.array([0.1, 0.1, 0.1])
        self.kd = np.array([0.5, 0.5, 0.1])

        # Deadband & saturation
        self.epsilon = np.array([0.0005, 0.0005, 0.0005])  # 2 mm
        self.max_speed = 0.22       # m/s per axis (conservative for Lite 6)

        # Home position (edit to your preferred safe home)
        self.home_position = np.array([0.227, 0.00, 0.468])

        # Center initialized from current EE pose when TF becomes available
        self.center = None

        # Timing / derivative memory
        self.start_time = self.get_clock().now()
        self.prev_error = np.zeros(3)
        self.prev_time = self.get_clock().now()

        # Log throttle
        self.last_info_time = self.get_clock().now()

        # Keyboard
        self._start_keyboard()

        # 50 Hz loop
        self.timer = self.create_timer(0.01, self._loop)

        self.get_logger().info(
            "CircleServoXArmLite6 initialized. 'p'=pause/resume, 'h'=home."
        )

    # ---------------------------
    # Keyboard controls
    # ---------------------------
    def _start_keyboard(self):
        def on_press(key):
            if hasattr(key, "char") and key.char == "p":
                if self.robot_state == RobotState.RUNNING:
                    self.robot_state = RobotState.PAUSED
                    self._publish_zero()
                    self.get_logger().warn("🔴 Paused.")
                elif self.robot_state == RobotState.PAUSED:
                    self.robot_state = RobotState.RUNNING
                    # reset derivative history to avoid a dt spike
                    self.prev_time = self.get_clock().now()
                    self.prev_error = np.zeros(3)
                    self.get_logger().info("🟢 Resumed.")

            if hasattr(key, "char") and key.char == "h":
                self.robot_state = RobotState.HOME
                self.get_logger().info("🔄 Going HOME...")

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

    # ---------------------------
    # TF pose read (position only)
    # ---------------------------
    def _read_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "link_base", "link_eef", rclpy.time.Time()
            )
            return np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ], dtype=float)
        except Exception as e:
            now = self.get_clock().now()
            if (now - self.last_info_time).nanoseconds > 2e9:
                self.get_logger().warn(f"TF not ready: {e}")
                self.last_info_time = now
            return None

    # ---------------------------
    # Circle target generator
    # ---------------------------
    def _circle_target(self, t_sec: float):
        cx, cy, cz = self.center
        w = 2.0 * math.pi * self.frequency

        # soft-start ramp (0 -> 1 in first 2 seconds)
        ramp = min(max(t_sec / 2.0, 0.0), 1.0)

        radius_x = ramp * self.radius
        radius_y = ramp * (self.radius / 2)
        
        a = radius_x * math.cos(w * t_sec)
        b = radius_y * math.sin(w * t_sec)

        if self.plane == "xy":
            x = cx + a
            y = cy + b
            z = cz if self.hold_z else cz
        elif self.plane == "xz":
            x = cx + a
            y = cy
            z = (cz + b) if not self.hold_z else cz
        elif self.plane == "yz":
            x = cx
            y = cy + a
            z = (cz + b) if not self.hold_z else cz
        else:
            x, y, z = cx + a, cy + b, cz

        return np.array([x, y, z], dtype=float)

    # ---------------------------
    # Publish TwistStamped
    # ---------------------------
    def _publish_twist(self, v_xyz: np.ndarray):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "link_base"
        cmd.twist.linear.x = float(v_xyz[0])
        cmd.twist.linear.y = float(v_xyz[1])
        cmd.twist.linear.z = float(v_xyz[2])
        cmd.twist.angular.x = 0.0
        cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = 0.0
        self.servo_pub.publish(cmd)

    def _publish_zero(self):
        self._publish_twist(np.zeros(3))

    # ---------------------------
    # PD servo step
    # ---------------------------
    def _servo_to(self, target_pos: np.ndarray):
        current = self._read_pose()
        if current is None:
            return

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 1e-6

        error = target_pos - current
        d_error = (error - self.prev_error) / dt
        
        # Tiempo actual desde inicio del círculo
        t_global = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        self.csv_writer.writerow([
            t_global, target_pos[0], target_pos[1], target_pos[2],
            current[0], current[1], current[2],
            error[0], error[1], error[2]
        ])
        self.csv_file.flush()
        if t_global > self.rmse_start_delay:
            self.rmse_sum += error**2
            self.rmse_count += 1

        active_mask = np.abs(error) > self.epsilon
        v = self.kp * error + self.kd * d_error
        v = np.where(active_mask, v, 0.0)

        # clip per-axis speed
        v = np.clip(v, -self.max_speed, self.max_speed)

        self._publish_twist(v)

        self.prev_error = error
        self.prev_time = now

        # log @ 1 Hz
        if (now - self.last_info_time).nanoseconds > 1e9:
            self.get_logger().info(
                f"pos={current.round(3)} target={target_pos.round(3)} vel={v.round(3)}"
            )
            if self.rmse_count > 0:
                rmse = np.sqrt(self.rmse_sum / self.rmse_count)
                rmse_total = np.sqrt(np.sum(self.rmse_sum) / self.rmse_count)
                self.get_logger().info(
                    f"RMSE -> X: {rmse[0]:.6f} m | "
                    f"Y: {rmse[1]:.6f} m | "
                    f"Z: {rmse[2]:.6f} m |"
                    f"TOTAL: {rmse_total:.6f} m"
                )
            self.last_info_time = now

    # ---------------------------
    # Main loop / state machine
    # ---------------------------
    def _loop(self):
        # Initialize circle center from current EE pose
        if self.center is None:
            p = self._read_pose()
            if p is None:
                return
            self.center = p.copy()
            self.rmse_sum = np.zeros(3)
            self.rmse_count = 0
            self.start_time = self.get_clock().now()
            self.prev_time = self.get_clock().now()
            self.prev_error = np.zeros(3)
            self.get_logger().info(f"✅ Center set to {self.center.round(3)}")
            return

        if self.robot_state == RobotState.PAUSED:
            return

        if self.robot_state == RobotState.HOME:
            current = self._read_pose()
            if current is None:
                return
            error = self.home_position - current
            dist = float(np.linalg.norm(error))
            if dist < 0.005:
                self._publish_zero()
                self.robot_state = RobotState.RUNNING
                # re-center circle at current pose after homing
                self.center = current.copy()
                self.start_time = self.get_clock().now()
                self.prev_time = self.get_clock().now()
                self.prev_error = np.zeros(3)
                self.get_logger().info("✅ Home reached. Circle re-centered and resumed.")
                return

            direction = np.where(np.abs(error) > 1e-4, np.sign(error), 0.0)
            v = direction * 0.08  # even gentler homing speed
            v = np.clip(v, -self.max_speed, self.max_speed)
            self._publish_twist(v)
            return

        # RUNNING: generate and track circular target
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        target = self._circle_target(t)
        self._servo_to(target)


def main(args=None):
    rclpy.init(args=args)
    node = CircleServoXArmLite6()
    try:
        rclpy.spin(node)
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



