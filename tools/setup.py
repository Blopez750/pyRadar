"""
setup.py — Cross-platform setup for the Stingray X-Band radar project.

Usage:
    Linux:   bash setup.sh          (installs apt packages first, then calls this)
             python3 setup.py       (if apt packages are already installed)
    Windows: python setup.py
"""

import glob
import os
import platform
import subprocess
import sys
import venv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REQUIREMENTS = os.path.join(PROJECT_DIR, "requirements.txt")
VENV_DIR = os.path.join(PROJECT_DIR, ".venv")
IS_WINDOWS = platform.system() == "Windows"

# Packages that need special handling per platform
SKIP_LINUX = {"pyqt5", "pyqt5-qt5", "pyqt5-sip", "pyqtgraph", "pywin32"}
SKIP_WINDOWS = set()  # All pip-installable on Windows


def run(cmd, **kwargs):
    print(f"  > {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)


def get_venv_python():
    if IS_WINDOWS:
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def get_venv_pip():
    if IS_WINDOWS:
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip")


def create_venv():
    if os.path.isdir(VENV_DIR):
        print(f"Virtual environment already exists: {VENV_DIR}")
    else:
        print(f"Creating virtual environment: {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    # Upgrade pip
    run(f'"{get_venv_python()}" -m pip install --upgrade pip')


def filtered_requirements(skip_set):
    """Return requirements lines, excluding packages in skip_set."""
    lines = []
    with open(REQUIREMENTS) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Extract package name (before ==, @, etc.)
            name = stripped.split("==")[0].split("@")[0].split(">=")[0].strip()
            if name.lower().replace("-", "").replace("_", "") not in {
                s.lower().replace("-", "").replace("_", "") for s in skip_set
            }:
                lines.append(stripped)
    return lines


def install_requirements():
    skip = SKIP_LINUX if not IS_WINDOWS else SKIP_WINDOWS
    reqs = filtered_requirements(skip)

    # Write a temporary filtered requirements file
    tmp_req = os.path.join(PROJECT_DIR, ".tmp_requirements.txt")
    with open(tmp_req, "w") as f:
        f.write("\n".join(reqs) + "\n")

    try:
        print(f"\nInstalling {len(reqs)} Python packages...")
        run(f'"{get_venv_pip()}" install -r "{tmp_req}"')
    finally:
        os.remove(tmp_req)


def link_system_pyqt_linux():
    """Symlink system-installed PyQt5/pyqtgraph into the venv (Linux/ARM only)."""
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_sp = os.path.join(VENV_DIR, "lib", py_ver, "site-packages")
    sys_sp = "/usr/lib/python3/dist-packages"

    if not os.path.isdir(sys_sp):
        print("Warning: system site-packages not found, skipping PyQt5 symlink")
        return

    # Link main package dirs + metadata
    for pattern in ["PyQt5", "PyQt5*", "pyqtgraph", "pyqtgraph*", "sip*"]:
        for src in glob.glob(os.path.join(sys_sp, pattern)):
            name = os.path.basename(src)
            dst = os.path.join(venv_sp, name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
                print(f"  Linked {name}")


def verify():
    """Quick smoke test: import key packages."""
    print("\nVerifying imports...")
    test_code = (
        "import numpy, scipy, pandas, matplotlib, h5py; "
        "import iio, adi; "
        "print('All core imports OK')"
    )
    try:
        run(f'"{get_venv_python()}" -c "{test_code}"')
    except subprocess.CalledProcessError:
        print("Warning: Some imports failed. Check the output above.")


def print_activate_hint():
    if IS_WINDOWS:
        activate = os.path.join(VENV_DIR, "Scripts", "activate.bat")
        print(f"\nActivate with:  {activate}")
        print(f"  Or PowerShell: {VENV_DIR}\\Scripts\\Activate.ps1")
    else:
        activate = os.path.join(VENV_DIR, "bin", "activate")
        print(f"\nActivate with:  source {activate}")


def check_libiio_windows():
    """On Windows, check for libiio DLL and guide the user if missing."""
    try:
        subprocess.check_call(
            [get_venv_python(), "-c", "import iio"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("\n" + "=" * 60)
        print("WARNING: libiio not found on this system.")
        print("Please install the libiio Windows installer from:")
        print("  https://github.com/analogdevicesinc/libiio/releases")
        print("Download the .exe installer, run it, and ensure the DLL")
        print("directory is on your system PATH.")
        print("=" * 60)


def main():
    print("=" * 60)
    print(f"  Stingray X-Band Radar — Setup ({platform.system()})")
    print("=" * 60)

    create_venv()
    install_requirements()

    if IS_WINDOWS:
        check_libiio_windows()
    else:
        link_system_pyqt_linux()

    verify()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print_activate_hint()


if __name__ == "__main__":
    main()
