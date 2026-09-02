# Laptop Robot Client requirements

- Ubuntu 22.04 preferred; wired isolated LAN with fixed/private addressing.
- Robot drivers, camera capture, hard safety checks, watchdog, action queue, and local safe-stop must run on the laptop.
- Policy server defaults to `127.0.0.1`; LAN binding must never expose the gRPC port to the public internet.
- Configure host firewall to allow only the desktop's private IP and selected port.
- On timeout, disconnect, NaN/non-finite action, malformed dimension, stale observation, or empty queue: hold a validated safe pose or invoke the vendor-defined stop. Never repeat an unknown action.
- Record monotonic timestamps, round-trip latency, queue depth, dropped/stale messages, and safety transitions.

