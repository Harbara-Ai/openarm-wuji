# Real robot migration checklist

- Obtain exact customized arm model/version, URDF/MJCF, joint names/order/units/limits, actuator and encoder mapping.
- Confirm control mode, command/watchdog timing, CAN/serial/EtherCAT interface, emergency stop, brakes, and homing procedure.
- Obtain Wuji Hand variant, side, firmware/API version, calibration, joint order/direction/range, and thermal/current limits.
- Measure arm tool frame to hand base transform and validate axes, scale, handedness, collision geometry, and payload/inertia.
- Benchmark the low-spec laptop camera timestamping, control jitter, network latency, packet loss, and safe-stop behavior.
- Keep hardware enable behind an explicit configuration flag and perform unloaded, low-speed commissioning first.

