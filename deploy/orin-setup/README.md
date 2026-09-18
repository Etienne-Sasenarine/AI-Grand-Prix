# Orin / Betaflight bench setup

Status: Windows OpenSSH is installed. User-provided specification identifies Orin NX 16GB, Seeed A603, JetPack 6.2, and UART to the Betaflight FC. Actual device identity, login, UART mapping/baud/protocol, and firmware version remain to be verified. No motor commands have been sent.

For the documented A603, connect its W6 Micro-USB device port to the PC with a data cable and boot normally using its appropriate power supply. Seeed documents SSH at 192.168.55.1 on JetPack 6.2. Do not enter recovery mode. Reference: https://wiki.seeedstudio.com/headless_setup_and_recovery_for_a603/

The supplied specification mentions existing FC serial demo applications. Inspect those before selecting a UART device or baud rate. UART alone does not establish that MSP is enabled or identify the application protocol.

Verified on the Orin: SSH as dcl using the dedicated PC key ~/.ssh/orin_dcl_setup; Ubuntu 22.04.5, L4T 36.4.3, pyserial installed, dcl in dialout. Existing code is in /home/dcl/target/msp. Its documentation and historical sign-off identify /dev/ttyTHS1 at 115200 baud with MSP, and historically BTFL 4.4.3 / board SH74 (not yet verified live). The supplied demo arms the FC and runs a control profile; do not use it as an individual motor bench test.

Windows USB DHCP was observed issuing 15-second leases and losing the address. fix-usb-network.ps1 sets 192.168.55.100/24 on the specifically identified NVIDIA RNDIS interface for the current Windows session, without a gateway. SSH access then succeeded. The Orin remains on a separate power supply; the operator has been told to keep the flight battery disconnected.

## Completed motor test

After the operator confirmed props removed and powered the drone, live MSP confirmed BTFL 4.4.3 / API 1.45, board SH74, four DShot300 motors, 3D disabled, and disarmed status. Installed motor_test.py on the Orin as /home/dcl/target/msp/codex_motor_test.py. Default invocation only inspects; --run --props-off runs four individual one-second tests at external output 1050. The test leaves an MSP arming block enabled until the FC reboots. It does not save configuration or send RC arming commands.

Successful physical test, confirmed through RPM telemetry:

| Motor | Peak RPM | RPM after stop |
|---|---:|---:|
| 1 | 1328 | 0 |
| 2 | 1342 | 0 |
| 3 | 1342 | 0 |
| 4 | 1385 | 0 |

Final stop commands acknowledged and all motor outputs read back as 1000. A simulated telemetry-failure test also verified that the script aborts and sends final stop commands. The operator was instructed to disconnect the flight battery after completion.

Read-only check from this PC:

```powershell
ssh -i "$env:USERPROFILE\.ssh\orin_dcl_setup" -o IdentitiesOnly=yes dcl@192.168.55.1 'python3 /home/dcl/target/msp/codex_motor_test.py'
```

To repeat a bench test, physically remove all propellers, secure the frame, keep clear, power the ESCs, then append `--run --props-off` inside the quoted remote command. Add `--motor 1` to test just motor 1. This script is for this verified hardware/configuration only. Its stop handling cannot protect against broken UART wiring or Orin power loss; disconnect the flight battery if a motor fails to stop. Disconnect the flight battery after testing. Never use this for flight.

Remove all propellers and leave the ESC/flight battery disconnected during connection setup. Power the Orin using the supply specified for its carrier board. The PC USB connection must use a supported device/OTG port and a data cable; an arbitrary USB host port is not suitable. Ethernet on the same LAN is an alternative.

`check-connection.ps1` checks the common Jetson USB-network address, or an address supplied with `-Address`. A reachable port does not establish device identity.

`inspect-orin.sh` is a read-only inventory to run on the Orin once SSH identity and authentication are established. It does not open FC serial devices, change settings, install packages, or start motors.

After connection: identify the FC and its Betaflight version, inspect existing configuration and MSP transport, then prepare a bounded, low-output individual-motor test with a stop command appropriate to that firmware and ESC configuration. Verify props are removed and the bench is clear before powering ESCs. Do not assume a disconnected PC automatically stops a motor test.

Reference: https://betaflight.com/docs/wiki/app/motors-tab
