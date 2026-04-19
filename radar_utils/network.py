"""Temporary FPGA network configuration.

Adds 192.168.0.100/24 as a secondary address on the Ethernet interface so the
host can reach the FPGA at 192.168.0.101.  The address is non-persistent — it
disappears on reboot, leaving the port free for DHCP / internet.
"""

import platform
import subprocess

FPGA_ADDR = "192.168.0.100"
FPGA_MASK = "255.255.255.0"
LINUX_IFACE = "enP2p1s0"


def configure_fpga_network():
    """Add a temporary IP alias for the FPGA link (Linux or Windows)."""
    if platform.system() == "Linux":
        _configure_linux()
    elif platform.system() == "Windows":
        _configure_windows()
    else:
        print(f"--> [WARN] Unsupported OS: {platform.system()}")
        print(f"    Manually assign {FPGA_ADDR}/24 to your Ethernet adapter.")


def _configure_linux():
    result = subprocess.run(["ip", "addr", "show", "dev", LINUX_IFACE],
                            capture_output=True, text=True)
    if FPGA_ADDR not in result.stdout:
        subprocess.run(["sudo", "ip", "addr", "add",
                        f"{FPGA_ADDR}/24", "dev", LINUX_IFACE], check=True)
        subprocess.run(["sudo", "ip", "link", "set", LINUX_IFACE, "up"],
                        check=True)
        print(f"--> Added temporary {FPGA_ADDR}/24 on {LINUX_IFACE}")
    else:
        print(f"--> {FPGA_ADDR}/24 already present on {LINUX_IFACE}")


def _configure_windows():
    result = subprocess.run(["netsh", "interface", "ip", "show", "address"],
                            capture_output=True, text=True)
    if FPGA_ADDR in result.stdout:
        print(f"--> {FPGA_ADDR} already configured")
        return

    iface_result = subprocess.run(
        ["netsh", "interface", "show", "interface"],
        capture_output=True, text=True)
    iface = None
    for line in iface_result.stdout.splitlines():
        if "Connected" in line and "Ethernet" in line:
            iface = line.split()[-1]
            break
    if iface is None:
        print("--> [WARN] No connected Ethernet adapter found.")
        print(f"    Run manually: netsh interface ip add address \"Ethernet\" {FPGA_ADDR} {FPGA_MASK}")
        return

    subprocess.run([
        "netsh", "interface", "ip", "add", "address",
        iface, FPGA_ADDR, FPGA_MASK
    ], check=True)
    print(f"--> Added temporary {FPGA_ADDR} on {iface}")


def ensure_fpga_network():
    """Call at startup — configures network, prints a warning on failure."""
    try:
        configure_fpga_network()
    except Exception as e:
        print(f"--> [WARN] Could not configure FPGA network: {e}")
        if platform.system() == "Windows":
            print(f"    Run as Administrator: netsh interface ip add address \"Ethernet\" {FPGA_ADDR} {FPGA_MASK}")
        else:
            print(f"    Run manually: sudo ip addr add {FPGA_ADDR}/24 dev {LINUX_IFACE}")
