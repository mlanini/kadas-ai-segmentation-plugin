import subprocess
import sys
import os
import shutil
import platform
import tempfile
import time
import re
import hashlib
from typing import Tuple, Optional, Callable, List

from qgis.core import QgsMessageLog, Qgis

from .model_config import SAM_PACKAGE, TORCH_MIN, TORCHVISION_MIN, USE_SAM2


PLUGIN_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PLUGIN_ROOT_DIR  # src/ directory
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"

# Support custom cache directory via environment variable (for corporate environments)
# AI_SEGMENTATION_CACHE_DIR allows IT admins to specify a directory with write permissions
CACHE_DIR = os.environ.get("AI_SEGMENTATION_CACHE_DIR")
if not CACHE_DIR:
    # Default locations if environment variable not set
    if sys.platform == "win32":
        CACHE_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "kadas_ai_segmentation")
    else:
        CACHE_DIR = os.path.expanduser("~/.kadas_ai_segmentation")

VENV_DIR = os.path.join(CACHE_DIR, f'venv_{PYTHON_VERSION}')
LIBS_DIR = os.path.join(PLUGIN_ROOT_DIR, 'libs')

REQUIRED_PACKAGES = [
    ("numpy", ">=1.26.0,<2.0.0"),
    ("torch", TORCH_MIN),
    ("torchvision", TORCHVISION_MIN),
    SAM_PACKAGE,
    ("pandas", ">=1.3.0"),
    ("rasterio", ">=1.3.0"),
]

DEPS_HASH_FILE = os.path.join(VENV_DIR, "deps_hash.txt")
CUDA_FLAG_FILE = os.path.join(VENV_DIR, "cuda_installed.txt")

# Bump this when install logic changes significantly (e.g., --no-cache-dir,
# new retry strategies) to force a dependency re-install on plugin update.
# This invalidates the deps hash so users with stale cuda_fallback flags
# get a clean retry with the improved install logic.
_INSTALL_LOGIC_VERSION = "3"

# Bumped independently of _INSTALL_LOGIC_VERSION so only users with a stale
# cuda_fallback flag get a targeted CUDA retry — without forcing a full
# dependency reinstall for everyone.  Increment this whenever the CUDA
# install logic is fixed in a way that makes a previous fallback worth
# retrying (e.g. adding --force-reinstall to overcome pip version skipping).
_CUDA_LOGIC_VERSION = "3"


def _log(message: str, level=Qgis.Info):
    QgsMessageLog.logMessage(message, "AI Segmentation", level=level)


def _get_permission_error_help() -> str:
    """Get platform-specific instructions for setting AI_SEGMENTATION_CACHE_DIR.
    
    Provides actionable guidance for users in corporate environments with restricted
    folder permissions. Instructions show how to set the environment variable on
    Windows, macOS, and Linux.
    
    Returns:
        Multi-line help message with platform-specific steps.
    """
    if sys.platform == "win32":
        return """Installation failed due to restricted folder permissions.

To install dependencies in a custom directory:
1. Open Windows System Settings
2. Search for "Environment Variables" and click "Edit environment variables for your account"
3. Click "New" under User variables
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: D:\\MyPlugins\\kadas_ai_segmentation
   (choose a folder where you have write permissions)
6. Click OK to save
7. Restart KADAS
8. Try installing again

If you need help choosing a suitable directory, contact your IT department.

Current restricted location: {}""".format(CACHE_DIR)
    
    elif sys.platform == "darwin":
        return """Installation failed due to restricted folder permissions.

To install dependencies in a custom directory:
1. Open Terminal (Applications > Utilities > Terminal)
2. Run this command:
   echo 'export AI_SEGMENTATION_CACHE_DIR="$HOME/MyPlugins/kadas_ai"' >> ~/.zshrc
3. Close Terminal and restart KADAS
4. Try installing again

Note: If you use bash instead of zsh, replace ~/.zshrc with ~/.bash_profile

Current restricted location: {}""".format(CACHE_DIR)
    
    else:  # Linux
        return """Installation failed due to restricted folder permissions.

To install dependencies in a custom directory:
1. Open Terminal
2. Run this command:
   echo 'export AI_SEGMENTATION_CACHE_DIR="$HOME/myplugins/kadas_ai"' >> ~/.bashrc
3. Close Terminal and restart KADAS
4. Try installing again

Note: If you use zsh, replace ~/.bashrc with ~/.zshrc

Current restricted location: {}""".format(CACHE_DIR)


def _write_cuda_flag(value: str):
    """Write CUDA flag to venv directory (e.g., 'cpu_only', 'cuda_fallback:3', 'cuda')."""
    os.makedirs(VENV_DIR, exist_ok=True)
    with open(CUDA_FLAG_FILE, "w", encoding="utf-8") as f:
        f.write(value)


def _read_cuda_flag() -> Optional[str]:
    """Read CUDA flag from venv directory."""
    if not os.path.exists(CUDA_FLAG_FILE):
        return None
    with open(CUDA_FLAG_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def _read_cuda_fallback_version() -> Optional[str]:
    """Read CUDA fallback version from cuda_installed.txt.
    Returns '1', '2', '3' etc. if cuda_fallback is versioned, else None."""
    flag = _read_cuda_flag()
    if flag and flag.startswith("cuda_fallback:"):
        return flag.split(":", 1)[1]
    return None


def needs_cuda_upgrade() -> bool:
    """Check if we should re-attempt CUDA installation.

    Re-attempts CUDA if:
    1. cuda_fallback flag exists (previous CUDA install failed), AND
    2. The fallback version is older than the current CUDA logic version

    This allows us to trigger a new CUDA attempt after plugin updates
    that fix CUDA installation issues, without forcing a full dependency
    reinstall for everyone.

    Returns:
        True if we should retry CUDA install (clears cuda_fallback flag)
        False otherwise
    """
    flag = _read_cuda_flag()
    if not flag or not flag.startswith("cuda_fallback:"):
        return False

    old_version = _read_cuda_fallback_version()
    if old_version is None:
        return False

    try:
        if int(old_version) < int(_CUDA_LOGIC_VERSION):
            _log(
                "CUDA logic updated (v{} -> v{}), will retry GPU acceleration".format(
                    old_version, _CUDA_LOGIC_VERSION
                ),
                Qgis.Info,
            )
            os.remove(CUDA_FLAG_FILE)
            return True
    except ValueError:
        pass

    return False


def _compute_deps_hash() -> str:
    """Compute MD5 hash of dependency specs + install logic version.
    If this changes, dependencies need to be reinstalled."""
    specs = [f"{name}{version}" for name, version in REQUIRED_PACKAGES]
    combined = "|".join(specs) + f"|logic_v{_INSTALL_LOGIC_VERSION}"
    return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()


def _read_deps_hash() -> Optional[str]:
    """Read stored deps hash from the venv directory."""
    if not os.path.exists(DEPS_HASH_FILE):
        return None
    with open(DEPS_HASH_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def _write_deps_hash():
    """Write the current deps hash to the venv directory."""
    os.makedirs(VENV_DIR, exist_ok=True)
    with open(DEPS_HASH_FILE, "w", encoding="utf-8") as f:
        f.write(_compute_deps_hash())


def _log_system_info():
    """Log system information for debugging installation issues."""
    try:
        kadas_version = Qgis.QGIS_VERSION
    except Exception:
        kadas_version = "Unknown"

    info_lines = [
        "=" * 50,
        "Installation Environment:",
        f"  OS: {sys.platform} ({platform.system()} {platform.release()})",
        f"  Architecture: {platform.machine()}",
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"  KADAS: {kadas_version}",
        "=" * 50,
    ]
    for line in info_lines:
        _log(line, Qgis.Info)


def _check_rosetta_warning() -> Optional[str]:
    """
    On macOS ARM, detect if running under Rosetta (x86_64 emulation).
    Returns warning message if Rosetta detected, None otherwise.
    """
    if sys.platform != "darwin":
        return None

    machine = platform.machine()

    # On Apple Silicon running native: machine = "arm64"
    # If running under Rosetta: machine = "x86_64" but CPU is Apple Silicon
    if machine == "x86_64":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if "Apple" in result.stdout:
                return (
                    "Warning: KADAS is running under Rosetta (x86_64 emulation) "
                    "on Apple Silicon. This may cause compatibility issues. "
                    "Consider using the native ARM64 version of KADAS for best performance."
                )
        except Exception:
            pass

    return None


def _needs_cu128(gpu_name: str) -> bool:
    """
    Detect if the GPU requires CUDA 12.8 wheels.
    RTX 50-series (Blackwell architecture, compute capability sm_120)
    needs cu128 because cu121/cu126 wheels don't include SM_120 kernels.
    """
    if not gpu_name:
        return False
    return "RTX 50" in gpu_name.upper()


def detect_nvidia_gpu() -> Tuple[bool, str]:
    """
    Detect if an NVIDIA GPU is present by querying nvidia-smi.
    Returns (True, "GPU Name") or (False, "").
    """
    try:
        subprocess_kwargs = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            subprocess_kwargs["startupinfo"] = startupinfo

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            **subprocess_kwargs,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0].strip()
            _log(f"NVIDIA GPU detected: {gpu_name}", Qgis.Info)
            return True, gpu_name
    except FileNotFoundError:
        pass  # nvidia-smi not found = no NVIDIA GPU
    except subprocess.TimeoutExpired:
        _log("nvidia-smi timed out", Qgis.Warning)
    except Exception as e:
        _log(f"nvidia-smi check failed: {e}", Qgis.Warning)

    return False, ""


def cleanup_old_venv_directories() -> List[str]:
    """
    Remove old venv_pyX.Y directories that don't match current Python version.
    Returns list of removed directories.
    """
    current_venv_name = f"venv_{PYTHON_VERSION}"
    removed = []

    try:
        for entry in os.listdir(SRC_DIR):
            if entry.startswith("venv_py") and entry != current_venv_name:
                old_path = os.path.join(SRC_DIR, entry)
                if os.path.isdir(old_path):
                    try:
                        shutil.rmtree(old_path)
                        _log(f"Cleaned up old venv directory: {entry}", Qgis.Info)
                        removed.append(entry)
                    except Exception as e:
                        _log(f"Failed to remove old venv {entry}: {e}", Qgis.Warning)
    except Exception as e:
        _log(f"Error scanning for old venv directories: {e}", Qgis.Warning)

    return removed


def _check_gdal_available() -> Tuple[bool, str]:
    """
    Check if GDAL system library is available (Linux only).
    Returns (is_available, help_message).
    """
    if sys.platform != "linux":
        return True, ""

    try:
        result = subprocess.run(
            ["gdal-config", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True, f"GDAL {result.stdout.strip()} found"
        return False, ""
    except FileNotFoundError:
        return False, (
            "GDAL library not found. Rasterio requires GDAL to be installed.\n"
            "Please install GDAL:\n"
            "  Ubuntu/Debian: sudo apt install libgdal-dev\n"
            "  Fedora: sudo dnf install gdal-devel\n"
            "  Arch: sudo pacman -S gdal"
        )
    except Exception:
        return True, ""  # Assume OK if check fails


_SSL_ERROR_PATTERNS = [
    "ssl",
    "certificate verify failed",
    "CERTIFICATE_VERIFY_FAILED",
    "SSLError",
    "SSLCertVerificationError",
    "tlsv1 alert",
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",
]


def _is_ssl_error(stderr: str) -> bool:
    """Detect SSL/certificate errors in pip output."""
    stderr_lower = stderr.lower()
    return any(pattern.lower() in stderr_lower for pattern in _SSL_ERROR_PATTERNS)


def _get_pip_ssl_flags() -> List[str]:
    """Get pip flags to bypass SSL verification for corporate proxies."""
    return [
        "--trusted-host", "pypi.org",
        "--trusted-host", "pypi.python.org",
        "--trusted-host", "files.pythonhosted.org",
    ]


def _get_ssl_error_help() -> str:
    """Get actionable help message for persistent SSL errors."""
    return (
        "SSL certificate verification failed. "
        "This is usually caused by a corporate proxy or firewall "
        "intercepting HTTPS connections.\n\n"
        "Please try:\n"
        "  1. Ask your IT team to whitelist: pypi.org, "
        "pypi.python.org, files.pythonhosted.org\n"
        "  2. Check your proxy settings in KADAS "
        "(Settings > Options > Network)\n"
        "  3. If using a VPN, try disconnecting temporarily"
    )


def _is_antivirus_error(stderr: str) -> bool:
    """Detect antivirus/permission blocking in pip output."""
    stderr_lower = stderr.lower()
    patterns = [
        "access is denied",
        "winerror 5",
        "winerror 225",
        "permission denied",
        "operation did not complete successfully because the file contains a virus",
        "blocked by your administrator",
        "blocked by group policy",
    ]
    return any(p in stderr_lower for p in patterns)


def _get_pip_antivirus_help(venv_dir: str) -> str:
    """Get actionable help message for antivirus blocking pip."""
    return (
        "Installation was blocked, likely by antivirus software "
        "or security policy.\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        "     {}\n"
        "  3. Run KADAS as administrator (right-click > Run as administrator)\n"
        "  4. Try the installation again"
    ).format(venv_dir)


# Windows NTSTATUS crash codes (both signed and unsigned representations)
_WINDOWS_CRASH_CODES = {
    3221225477,   # 0xC0000005 unsigned - ACCESS_VIOLATION
    -1073741819,  # 0xC0000005 signed   - ACCESS_VIOLATION
    3221225725,   # 0xC00000FD unsigned - STACK_OVERFLOW
    -1073741571,  # 0xC00000FD signed   - STACK_OVERFLOW
    3221225781,   # 0xC0000135 unsigned - DLL_NOT_FOUND
    -1073741515,  # 0xC0000135 signed   - DLL_NOT_FOUND
}


def _is_windows_process_crash(returncode: int) -> bool:
    """Detect Windows process crashes (ACCESS_VIOLATION, STACK_OVERFLOW, etc.)."""
    if sys.platform != "win32":
        return False
    return returncode in _WINDOWS_CRASH_CODES


def _get_crash_help(venv_dir: str) -> str:
    """Get actionable help for Windows process crash during pip install."""
    return (
        "The installer process crashed unexpectedly (access violation).\n\n"
        "This is usually caused by:\n"
        "  - Antivirus software (Windows Defender, etc.) blocking pip\n"
        "  - Corrupted virtual environment\n\n"
        "Please try:\n"
        "  1. Temporarily disable real-time antivirus scanning\n"
        "  2. Add an exclusion for the plugin folder:\n"
        "     {}\n"
        "  3. Click 'Reinstall Dependencies' to recreate the environment\n"
        "  4. If the issue persists, run KADAS as administrator"
    ).format(venv_dir)


def cleanup_old_venv_directories():
    """Remove old venv directories that were created inside the plugin folder.

    This is called on plugin startup to clean up legacy installations
    that used the old venv location (inside the plugin directory).
    New installations use ~/.kadas_ai_segmentation/venv_py3.x/
    """
    try:
        old_venv_pattern = os.path.join(PLUGIN_ROOT_DIR, "venv_py*")
        import glob
        for old_venv in glob.glob(old_venv_pattern):
            if os.path.isdir(old_venv):
                _log(f"Removing old venv directory: {old_venv}", Qgis.Info)
                shutil.rmtree(old_venv)
    except Exception as e:
        _log(f"Failed to cleanup old venv directories: {e}", Qgis.Warning)


def cleanup_old_libs():
    """Remove old libs/ directory from plugin folder.

    This is called on plugin startup to clean up legacy installations
    that used the libs/ directory for dependencies.
    """
    if os.path.exists(LIBS_DIR):
        try:
            _log(f"Removing old libs directory: {LIBS_DIR}", Qgis.Info)
            shutil.rmtree(LIBS_DIR)
        except Exception as e:
            _log(f"Failed to remove libs directory: {e}", Qgis.Warning)


def get_venv_dir() -> str:
    return VENV_DIR


def get_venv_site_packages(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Lib", "site-packages")
    else:
        # Detect actual Python version in venv (may differ from KADAS Python)
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            for entry in os.listdir(lib_dir):
                if entry.startswith("python") and os.path.isdir(os.path.join(lib_dir, entry)):
                    site_packages = os.path.join(lib_dir, entry, "site-packages")
                    if os.path.exists(site_packages):
                        return site_packages

        # Fallback to KADAS Python version (for new venv creation)
        py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return os.path.join(venv_dir, "lib", py_version, "site-packages")


def ensure_venv_packages_available():
    if not venv_exists():
        _log("Venv does not exist, cannot load packages", Qgis.Warning)
        return False

    site_packages = get_venv_site_packages()
    if not os.path.exists(site_packages):
        # Log detailed info for debugging
        venv_dir = get_venv_dir()
        lib_dir = os.path.join(venv_dir, "lib")
        if os.path.exists(lib_dir):
            contents = os.listdir(lib_dir)
            _log(f"Venv lib/ contents: {contents}", Qgis.Warning)
        _log(f"Venv site-packages not found: {site_packages}", Qgis.Warning)
        return False

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}", Qgis.Info)

    return True


def get_venv_python_path(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir, "bin", "python3")


def get_venv_pip_path(venv_dir: str = None) -> str:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
    
    # For manual folders, pip might not exist - return python -m pip instead
    if not os.path.exists(pip_path):
        return get_venv_python_path(venv_dir)  # Will use python -m pip
    
    return pip_path


def is_manual_venv_folder(venv_dir: str = None) -> bool:
    """Check if this is a manual folder (no pip.exe) created to bypass Group Policy."""
    if venv_dir is None:
        venv_dir = VENV_DIR
    
    # Check if pyvenv.cfg mentions "Manual folder"
    pyvenv_cfg = os.path.join(venv_dir, "pyvenv.cfg")
    if os.path.exists(pyvenv_cfg):
        try:
            with open(pyvenv_cfg, "r") as f:
                content = f.read()
                if "Manual folder" in content or "Group Policy bypass" in content:
                    return True
        except:
            pass
    
    # Also check if Scripts/pip.exe doesn't exist (manual folder indicator)
    pip_path = os.path.join(venv_dir, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(venv_dir, "bin", "pip")
    return not os.path.exists(pip_path)


def _get_kadas_python() -> Optional[str]:
    """
    Get the path to KADAS's bundled Python on Windows.

    KADAS ships with a signed Python interpreter. This is used as a fallback
    when the standalone Python download is blocked by anti-malware software.

    Returns the path to the Python executable, or None if not found/not Windows.
    """
    if sys.platform != "win32":
        return None

    # KADAS on Windows bundles Python under sys.prefix
    python_path = os.path.join(sys.prefix, "python.exe")
    if not os.path.exists(python_path):
        # Some KADAS installs place it under a python3 name
        python_path = os.path.join(sys.prefix, "python3.exe")

    if not os.path.exists(python_path):
        _log("KADAS bundled Python not found at sys.prefix", Qgis.Warning)
        return None

    # Verify it can execute
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(
            [python_path, "-c", "import sys; print(sys.version)"],
            capture_output=True, text=True, timeout=15,
            env=env, startupinfo=startupinfo,
        )
        if result.returncode == 0:
            _log(f"KADAS Python verified: {result.stdout.strip()}", Qgis.Info)
            return python_path
        else:
            _log(f"KADAS Python failed verification: {result.stderr}", Qgis.Warning)
            return None
    except Exception as e:
        _log(f"KADAS Python verification error: {e}", Qgis.Warning)
        return None


def _get_system_python() -> str:
    """
    Get the path to the Python executable for creating venvs.

    Uses the standalone Python downloaded by python_manager.
    On Windows, falls back to KADAS's bundled Python if standalone is unavailable
    (e.g. when anti-malware blocks the standalone download).
    """
    from .python_manager import standalone_python_exists, get_standalone_python_path

    if standalone_python_exists():
        python_path = get_standalone_python_path()
        _log(f"Using standalone Python: {python_path}", Qgis.Info)
        return python_path

    # On Windows, try KADAS's bundled Python as fallback
    if sys.platform == "win32":
        kadas_python = _get_kadas_python()
        if kadas_python:
            _log(
                "Standalone Python unavailable, using KADAS Python as fallback",
                Qgis.Warning
            )
            return kadas_python

    # No fallback available
    raise RuntimeError(
        "Python standalone not installed. "
        "Please click 'Install Dependencies' to download Python automatically."
    )


def venv_exists(venv_dir: str = None) -> bool:
    if venv_dir is None:
        venv_dir = VENV_DIR

    python_path = get_venv_python_path(venv_dir)
    return os.path.exists(python_path)


def create_venv(venv_dir: str = None, progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    # Log the cache directory location for user's reference
    if sys.platform == "win32":
        _log(f"Package cache directory: %APPDATA%\\kadas_ai_segmentation", Qgis.Info)
    _log(f"Creating virtual environment at: {venv_dir}", Qgis.Info)

    if progress_callback:
        progress_callback(10, "Creating virtual environment...")

    system_python = _get_system_python()
    _log(f"Using Python: {system_python}", Qgis.Info)

    # Try standard venv first
    cmd = [system_python, "-m", "venv", venv_dir]

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # Increased from 120s to 300s for slow corporate environments
                env=env,
                startupinfo=startupinfo,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # Increased from 120s to 300s for slow corporate environments
                env=env,
            )

        if result.returncode == 0:
            _log("Virtual environment created successfully", Qgis.Success)

            # Ensure pip is available (QGIS Python fallback may not include pip)
            pip_path = get_venv_pip_path(venv_dir)
            if not os.path.exists(pip_path):
                _log("pip not found in venv, bootstrapping with ensurepip...", Qgis.Info)
                python_in_venv = get_venv_python_path(venv_dir)
                ensurepip_cmd = [python_in_venv, "-m", "ensurepip", "--upgrade"]
                try:
                    ensurepip_result = subprocess.run(
                        ensurepip_cmd,
                        capture_output=True, text=True, timeout=300,  # Increased from 120s to 300s
                        env=env,
                        **({"startupinfo": startupinfo} if sys.platform == "win32" else {}),
                    )
                    if ensurepip_result.returncode == 0:
                        _log("pip bootstrapped via ensurepip", Qgis.Success)
                    else:
                        err = ensurepip_result.stderr or ensurepip_result.stdout
                        _log(f"ensurepip failed: {err[:200]}", Qgis.Warning)
                        return False, f"Failed to bootstrap pip: {err[:200]}"
                except Exception as e:
                    _log(f"ensurepip exception: {e}", Qgis.Warning)
                    return False, f"Failed to bootstrap pip: {str(e)[:200]}"

            if progress_callback:
                progress_callback(20, "Virtual environment created")
            return True, "Virtual environment created"
        else:
            error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
            
            # Check if venv creation was blocked by Group Policy (WinError 1260)
            if sys.platform == "win32" and ("1260" in error_msg or "gruppo" in error_msg.lower() or "group" in error_msg.lower()):
                _log("venv blocked by Group Policy, trying virtualenv fallback...", Qgis.Warning)
                return _create_venv_with_virtualenv(venv_dir, system_python, env, progress_callback)
            
            _log(f"Failed to create venv: {error_msg}", Qgis.Critical)
            return False, f"Failed to create venv: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log("Virtual environment creation timed out", Qgis.Critical)
        return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        _log(f"Python executable not found: {system_python}", Qgis.Critical)
        return False, f"Python not found: {system_python}"
    except PermissionError as e:
        # Corporate environment with restricted folder permissions (v0.6.5)
        error_msg = _get_permission_error_help()
        _log(error_msg, Qgis.Critical)
        return False, error_msg
    except OSError as e:
        # Check for permission-related OSError (some platforms raise OSError instead of PermissionError)
        error_str = str(e).lower()
        if "permission" in error_str or "denied" in error_str or "access" in error_str:
            error_msg = _get_permission_error_help()
            _log(error_msg, Qgis.Critical)
            return False, error_msg
        # Non-permission OSError - re-raise
        raise
    except Exception as e:
        error_str = str(e)
        # Check for Group Policy error in exception
        if sys.platform == "win32" and ("1260" in error_str or "gruppo" in error_str.lower() or "group" in error_str.lower()):
            _log("venv blocked by Group Policy, trying virtualenv fallback...", Qgis.Warning)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            return _create_venv_with_virtualenv(venv_dir, system_python, env, progress_callback)
        
        _log(f"Exception during venv creation: {error_str}", Qgis.Critical)
        return False, f"Error: {error_str[:200]}"


def _create_venv_with_virtualenv(venv_dir: str, system_python: str, env: dict, progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[bool, str]:
    """Fallback to virtualenv when venv is blocked by Group Policy.
    
    virtualenv doesn't use symlinks/hardlinks and works even with restricted permissions.
    """
    _log("Attempting to use virtualenv as fallback (corporate environment workaround)", Qgis.Info)
    
    if progress_callback:
        progress_callback(12, "Installing virtualenv tool...")
    
    # First, install virtualenv into QGIS Python if not available
    try:
        # Check if virtualenv is already available
        check_cmd = [system_python, "-m", "virtualenv", "--version"]
        
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            check_result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
                startupinfo=startupinfo
            )
        else:
            check_result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
        
        if check_result.returncode != 0:
            # Install virtualenv
            _log("virtualenv not found, installing...", Qgis.Info)
            install_cmd = [system_python, "-m", "pip", "install", "--user", "virtualenv"]
            
            if sys.platform == "win32":
                install_result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                    startupinfo=startupinfo
                )
            else:
                install_result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env
                )
            
            if install_result.returncode != 0:
                err = install_result.stderr or install_result.stdout or "Unknown error"
                
                # Check if pip itself is broken (OpenSSL, import errors, etc.)
                pip_broken = any(keyword in err for keyword in [
                    "RuntimeError", "ImportError", "ModuleNotFoundError",
                    "OpenSSL", "cryptography", "failed to load"
                ])
                
                if pip_broken:
                    _log("pip is broken (likely OpenSSL/crypto issue), skipping to manual folder fallback...", Qgis.Warning)
                    _log(f"pip error: {err[:500]}", Qgis.Warning)
                    return _create_manual_venv_folder(venv_dir, system_python, progress_callback)
                
                # Check if Group Policy is blocking pip install --user
                if "1260" in err or "Criteri di gruppo" in err or "Group Policy" in err:
                    _log("Group Policy blocks pip install --user, trying manual folder fallback...", Qgis.Warning)
                    return _create_manual_venv_folder(venv_dir, system_python, progress_callback)
                
                # Log full error for debugging
                _log(f"Failed to install virtualenv. Error output:\n{err}", Qgis.Critical)
                return False, f"Failed to install virtualenv tool. Check QGIS logs for details."
            
            _log("virtualenv installed successfully", Qgis.Success)
        
        if progress_callback:
            progress_callback(15, "Creating virtual environment with virtualenv...")
        
        # Create venv using virtualenv
        create_cmd = [system_python, "-m", "virtualenv", venv_dir, "--no-download"]
        
        if sys.platform == "win32":
            create_result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # Increased from 120s to 300s for slow corporate environments
                env=env,
                startupinfo=startupinfo
            )
        else:
            create_result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # Increased from 120s to 300s for slow corporate environments
                env=env
            )
        
        if create_result.returncode == 0:
            _log("Virtual environment created successfully with virtualenv", Qgis.Success)
            if progress_callback:
                progress_callback(20, "Virtual environment created")
            return True, "Virtual environment created (virtualenv)"
        else:
            err = create_result.stderr or create_result.stdout or "Unknown error"
            # Log full error for debugging
            _log(f"virtualenv creation failed. Full error:\n{err}", Qgis.Critical)
            
            # Check if pip/OpenSSL is broken - skip to manual folder
            pip_broken = any(keyword in err for keyword in [
                "RuntimeError", "ImportError", "ModuleNotFoundError",
                "OpenSSL", "cryptography", "failed to load"
            ])
            
            if pip_broken:
                _log("virtualenv failed due to pip/OpenSSL issue, trying manual folder fallback...", Qgis.Warning)
                return _create_manual_venv_folder(venv_dir, system_python, progress_callback)
            
            # Check if Group Policy is still blocking
            if "1260" in err or "Criteri di gruppo" in err or "Group Policy" in err:
                _log("virtualenv blocked by Group Policy, trying manual folder fallback...", Qgis.Warning)
                return _create_manual_venv_folder(venv_dir, system_python, progress_callback)
            
            return False, f"virtualenv failed to create environment. Check QGIS logs for details."
    
    except Exception as e:
        error_msg = str(e)
        _log(f"virtualenv fallback exception: {error_msg}", Qgis.Critical)
        
        # Check if Group Policy in exception - try manual folder fallback
        if "1260" in error_msg or "Criteri di gruppo" in error_msg or "Group Policy" in error_msg:
            _log("Both venv and virtualenv blocked by Group Policy, trying manual folder fallback...", Qgis.Warning)
            return _create_manual_venv_folder(venv_dir, system_python, progress_callback)
        
        return False, f"virtualenv fallback error. Check QGIS logs for details."


def _create_manual_venv_folder(venv_dir: str, system_python: str, progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[bool, str]:
    """Ultimate fallback: create a manual folder structure for packages without using venv/virtualenv.
    
    This bypasses Group Policy by not creating a "virtual environment" at all - just a folder
    where we'll install packages with pip --target.
    """
    _log("Creating manual package folder (Group Policy bypass strategy)", Qgis.Info)
    
    if progress_callback:
        progress_callback(15, "Creating manual package folder...")
    
    try:
        # Create directory structure
        os.makedirs(venv_dir, exist_ok=True)
        
        # Create Lib/site-packages directory (matches venv structure for compatibility)
        site_packages = os.path.join(venv_dir, "Lib", "site-packages")
        os.makedirs(site_packages, exist_ok=True)
        
        # Create Scripts directory for pip/executables
        scripts_dir = os.path.join(venv_dir, "Scripts") if sys.platform == "win32" else os.path.join(venv_dir, "bin")
        os.makedirs(scripts_dir, exist_ok=True)
        
        # Copy Python executable to Scripts folder (so we have a local Python)
        import shutil
        python_name = "python.exe" if sys.platform == "win32" else "python"
        python_copy = os.path.join(scripts_dir, python_name)
        
        try:
            shutil.copy2(system_python, python_copy)
            _log(f"Copied Python to: {python_copy}", Qgis.Info)
        except Exception as copy_err:
            # If we can't copy, we'll use system Python directly
            _log(f"Could not copy Python (will use system Python): {copy_err}", Qgis.Warning)
            python_copy = system_python
        
        # Create a pyvenv.cfg to mark this as a "venv-like" folder
        pyvenv_cfg = os.path.join(venv_dir, "pyvenv.cfg")
        with open(pyvenv_cfg, "w") as f:
            f.write(f"home = {os.path.dirname(system_python)}\n")
            f.write("include-system-site-packages = false\n")
            f.write(f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
            f.write("# Manual folder created to bypass Group Policy\n")
        
        _log(f"Manual package folder created: {venv_dir}", Qgis.Success)
        
        if progress_callback:
            progress_callback(20, "Manual package folder created")
        
        return True, "Manual package folder created (Group Policy bypass)"
    
    except Exception as e:
        _log(f"Failed to create manual package folder: {str(e)}", Qgis.Critical)
        return False, f"Could not create package folder: {str(e)}"


def install_dependencies(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment does not exist"

    # Check if this is a manual folder (Group Policy bypass)
    is_manual = is_manual_venv_folder(venv_dir)
    
    pip_path = get_venv_pip_path(venv_dir)
    python_path = get_venv_python_path(venv_dir)
    
    if is_manual:
        _log(f"Installing to manual folder (Group Policy bypass): {venv_dir}", Qgis.Info)
        _log(f"Using system Python for installation: {python_path}", Qgis.Info)
    else:
        _log(f"Installing dependencies using: {pip_path}", Qgis.Info)
    
    if cuda_enabled:
        _log("CUDA mode enabled - will install GPU-accelerated PyTorch", Qgis.Info)

    total_packages = len(REQUIRED_PACKAGES)
    base_progress = 20
    progress_per_package = 80 // total_packages

    python_path = get_venv_python_path(venv_dir)

    for i, (package_name, version_spec) in enumerate(REQUIRED_PACKAGES):
        if cancel_check and cancel_check():
            _log("Installation cancelled by user", Qgis.Warning)
            return False, "Installation cancelled"

        package_spec = f"{package_name}{version_spec}"
        current_progress = base_progress + (i * progress_per_package)

        is_cuda_package = cuda_enabled and package_name in ("torch", "torchvision")

        if progress_callback:
            if package_name == "torch" and cuda_enabled:
                progress_callback(
                    current_progress,
                    "Installing {} (CUDA ~2.5GB)... ({}/{})".format(
                        package_name, i + 1, total_packages))
            elif package_name == "torch":
                progress_callback(
                    current_progress,
                    "Installing {} (~600MB)... ({}/{})".format(
                        package_name, i + 1, total_packages))
            else:
                progress_callback(
                    current_progress,
                    "Installing {}... ({}/{})".format(
                        package_name, i + 1, total_packages))

        _log(f"[{i + 1}/{total_packages}] Installing {package_spec}...", Qgis.Info)

        pip_args = [
            "install",
            "--upgrade",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            "--prefer-binary",  # Prefer pre-built wheels to avoid C extension build issues
        ]
        
        # For manual folders, use --target to install directly to site-packages
        if is_manual:
            site_packages = os.path.join(venv_dir, "Lib", "site-packages")
            pip_args.extend(["--target", site_packages])
            _log(f"Using --target to install to: {site_packages}", Qgis.Info)
        
        pip_args.extend(_get_pip_proxy_args())
        pip_args.append(package_spec)

        # For CUDA-enabled torch/torchvision, use PyTorch's CUDA index
        if is_cuda_package:
            _, gpu_name = detect_nvidia_gpu()
            if _needs_cu128(gpu_name):
                cuda_index = "cu128"
            else:
                cuda_index = "cu121"
            pip_args.extend([
                "--index-url", "https://download.pytorch.org/whl/{}".format(cuda_index)
            ])
            _log("Using CUDA {} index for {}".format(cuda_index, package_name), Qgis.Info)

        # Use clean env to avoid QGIS PYTHONPATH/PYTHONHOME interference
        env = _get_clean_env_for_venv()

        subprocess_kwargs = _get_subprocess_kwargs()

        # Set appropriate timeouts based on package size and network conditions
        # Increased timeouts to handle slow corporate networks/proxies (v0.6.5)
        if is_cuda_package and package_name in ("torch", "torchvision"):
            pkg_timeout = 2400  # 40 min for CUDA wheels (~2.5GB)
        elif package_name == "torch":
            pkg_timeout = 3600  # 60 min for CPU torch (~600MB, slow on corporate networks)
        elif package_name == "torchvision":
            pkg_timeout = 1200  # 20 min for CPU torchvision
        else:
            pkg_timeout = 600  # 10 min for standard packages

        install_failed = False
        install_error_msg = ""
        last_returncode = None

        try:
            # Use python -m pip (more reliable than pip.exe on Windows
            # where antivirus may block the standalone pip executable)
            base_cmd = [python_path, "-m", "pip"] + pip_args

            # First attempt: standard pip install
            result = subprocess.run(
                base_cmd,
                capture_output=True, text=True, timeout=pkg_timeout,
                env=env, **subprocess_kwargs,
            )

            # If Windows process crash, retry with pip.exe as fallback
            if _is_windows_process_crash(result.returncode):
                _log(
                    "Process crash detected (code {}), "
                    "retrying with pip.exe...".format(result.returncode),
                    Qgis.Warning
                )
                if progress_callback:
                    progress_callback(
                        current_progress,
                        "Retrying {}... ({}/{})".format(
                            package_name, i + 1, total_packages)
                    )

                fallback_cmd = [pip_path] + pip_args
                result = subprocess.run(
                    fallback_cmd,
                    capture_output=True, text=True, timeout=pkg_timeout,
                    env=env, **subprocess_kwargs,
                )

            # If failed, check for SSL errors and retry with --trusted-host
            if result.returncode != 0 and not _is_windows_process_crash(result.returncode):
                error_output = result.stderr or result.stdout or ""

                if _is_ssl_error(error_output):
                    _log(
                        "SSL error detected, retrying with --trusted-host flags...",
                        Qgis.Warning
                    )
                    if progress_callback:
                        progress_callback(
                            current_progress,
                            "SSL error, retrying {}... ({}/{})".format(
                                package_name, i + 1, total_packages)
                        )

                    ssl_cmd = base_cmd[:3] + _get_pip_ssl_flags() + base_cmd[3:]
                    result = subprocess.run(
                        ssl_cmd,
                        capture_output=True, text=True, timeout=pkg_timeout,
                        env=env, **subprocess_kwargs,
                    )

            if result.returncode == 0:
                _log(f"✓ Successfully installed {package_spec}", Qgis.Success)
                if progress_callback:
                    progress_callback(current_progress + progress_per_package, f"✓ {package_name} installed")
            else:
                error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
                _log(f"✗ Failed to install {package_spec}: {error_msg[:500]}", Qgis.Critical)
                install_failed = True
                install_error_msg = error_msg
                last_returncode = result.returncode
                
                # Check for Group Policy error (WinError 1260)
                if sys.platform == "win32" and ("1260" in error_msg or "gruppo" in error_msg.lower() or "group" in error_msg.lower()):
                    _log("Group Policy error detected during package installation", Qgis.Critical)
                    return False, f"GROUP_POLICY_ERROR: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            _log(f"Installation of {package_spec} timed out", Qgis.Critical)
            install_failed = True
            install_error_msg = f"Installation of {package_name} timed out"
        except Exception as e:
            _log(f"Exception during installation of {package_spec}: {str(e)}", Qgis.Critical)
            install_failed = True
            install_error_msg = f"Error installing {package_name}: {str(e)[:200]}"
            
            # Check for Group Policy error (WinError 1260)
            error_str = str(e)
            if sys.platform == "win32" and ("1260" in error_str or "gruppo" in error_str.lower() or "group" in error_str.lower()):
                _log("Group Policy error detected during package installation", Qgis.Critical)
                
                # If this happens in a manual folder, it means Python execution itself is blocked
                if is_manual:
                    error_msg = (
                        "CRITICAL: Group Policy is blocking Python execution entirely.\n\n"
                        "This corporate environment has severe restrictions that prevent:\n"
                        "  1. Creating virtual environments (venv blocked)\n"
                        "  2. Using virtualenv tool (pip blocked)\n"
                        "  3. Running Python.exe directly (execution blocked)\n\n"
                        "This plugin cannot work in this environment.\n\n"
                        "Please contact your IT administrator to:\n"
                        "  - Whitelist Python execution (python.exe)\n"
                        "  - Allow pip package installation\n"
                        "  - Enable virtual environment creation\n\n"
                        "Alternatively, use a personal computer without these restrictions."
                    )
                    return False, error_msg
                
                return False, f"GROUP_POLICY_ERROR: {install_error_msg}"

        # CUDA → CPU silent fallback: if CUDA install failed, retry with CPU wheel
        if install_failed and is_cuda_package:
            _log(
                "CUDA install of {} failed, falling back to CPU version...".format(package_name),
                Qgis.Warning
            )
            if progress_callback:
                progress_callback(
                    current_progress,
                    "CUDA failed, installing {} (CPU)...".format(package_name)
                )

            cpu_pip_args = [
                "install", "--upgrade", "--no-warn-script-location",
                "--disable-pip-version-check", "--prefer-binary",
                package_spec
            ]
            cpu_cmd = [python_path, "-m", "pip"] + cpu_pip_args
            try:
                cpu_result = subprocess.run(
                    cpu_cmd,
                    capture_output=True, text=True, timeout=600,
                    env=env, **subprocess_kwargs,
                )
                if cpu_result.returncode == 0:
                    _log("✓ Successfully installed {} (CPU version)".format(package_spec), Qgis.Success)
                    if progress_callback:
                        progress_callback(
                            current_progress + progress_per_package,
                            "✓ {} installed (CPU)".format(package_name)
                        )
                    install_failed = False
                else:
                    cpu_err = cpu_result.stderr or cpu_result.stdout or ""
                    install_error_msg = "CUDA and CPU install both failed for {}: {}".format(
                        package_name, cpu_err[:200])
            except subprocess.TimeoutExpired:
                install_error_msg = "CUDA and CPU install both timed out for {}".format(package_name)
            except Exception as e:
                install_error_msg = "CUDA and CPU install both failed for {}: {}".format(
                    package_name, str(e)[:200])

        if install_failed:
            # Check for Windows process crash
            if last_returncode is not None and _is_windows_process_crash(last_returncode):
                crash_help = _get_crash_help(venv_dir)
                _log(crash_help, Qgis.Warning)
                install_error_msg = "{}\n\n{}".format(
                    "Process crashed (code {})".format(last_returncode),
                    crash_help)
                return False, f"Failed to install {package_name}: {install_error_msg}"

            # Check for SSL errors
            if _is_ssl_error(install_error_msg):
                ssl_help = _get_ssl_error_help()
                _log(ssl_help, Qgis.Warning)
                install_error_msg = "{}\n\n{}".format(install_error_msg[:200], ssl_help)
                return False, f"Failed to install {package_name}: {install_error_msg}"

            # Check for antivirus blocking
            if _is_antivirus_error(install_error_msg):
                av_help = _get_pip_antivirus_help(venv_dir)
                _log(av_help, Qgis.Warning)
                install_error_msg = "{}\n\n{}".format(install_error_msg[:200], av_help)
                return False, f"Failed to install {package_name}: {install_error_msg}"

            # Check for GDAL issues on Linux when rasterio fails
            if package_name == "rasterio":
                gdal_ok, gdal_help = _check_gdal_available()
                if not gdal_ok and gdal_help:
                    _log(gdal_help, Qgis.Warning)
                    install_error_msg = "{}\n\n{}".format(install_error_msg[:200], gdal_help)
                    return False, f"Failed to install {package_name}: {install_error_msg}"

            return False, f"Failed to install {package_name}: {install_error_msg[:200]}"

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

    _log("=" * 50, Qgis.Success)
    _log("All dependencies installed successfully!", Qgis.Success)
    _log(f"Virtual environment: {venv_dir}", Qgis.Success)
    _log("=" * 50, Qgis.Success)

    return True, "All dependencies installed successfully"


def _get_qgis_proxy_settings() -> Optional[str]:
    """Read proxy configuration from QGIS settings.

    Returns a proxy URL string like 'http://user:pass@host:port'
    or None if proxy is not configured or disabled.
    """
    try:
        from qgis.core import QgsSettings
        from urllib.parse import quote as url_quote

        settings = QgsSettings()
        enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not enabled:
            return None

        host = settings.value("proxy/proxyHost", "", type=str)
        if not host:
            return None

        port = settings.value("proxy/proxyPort", "", type=str)
        user = settings.value("proxy/proxyUser", "", type=str)
        password = settings.value("proxy/proxyPassword", "", type=str)

        proxy_url = "http://"
        if user:
            proxy_url += url_quote(user, safe="")
            if password:
                proxy_url += ":" + url_quote(password, safe="")
            proxy_url += "@"
        proxy_url += host
        if port:
            proxy_url += ":{}".format(port)

        return proxy_url
    except Exception as e:
        _log("Could not read QGIS proxy settings: {}".format(e), Qgis.Warning)
        return None


def _get_pip_proxy_args() -> List[str]:
    """Get pip --proxy argument if QGIS proxy is configured."""
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        _log("Using QGIS proxy for pip: {}".format(
            proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url),
            Qgis.Info
        )
        return ["--proxy", proxy_url]
    return []


def _get_clean_env_for_venv() -> dict:
    env = os.environ.copy()

    vars_to_remove = [
        'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',
        'QGIS_PREFIX_PATH', 'QGIS_PLUGINPATH',
    ]
    for var in vars_to_remove:
        env.pop(var, None)

    env["PYTHONIOENCODING"] = "utf-8"

    # Propagate QGIS proxy settings to environment for pip/network calls
    proxy_url = _get_qgis_proxy_settings()
    if proxy_url:
        env.setdefault("HTTP_PROXY", proxy_url)
        env.setdefault("HTTPS_PROXY", proxy_url)

    return env


def _get_subprocess_kwargs() -> dict:
    kwargs = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _get_verification_code(package_name: str) -> str:
    """
    Get verification code that actually TESTS the package works, not just imports.

    This catches issues like pandas C extensions not being built properly.
    """
    if package_name == "pandas":
        # Test that pandas C extensions work by creating a DataFrame
        return "import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]}); print(df.sum())"
    elif package_name == "numpy":
        # Test numpy array operations
        return "import numpy as np; a = np.array([1, 2, 3]); print(np.sum(a))"
    elif package_name == "torch":
        # Test torch tensor creation
        return "import torch; t = torch.tensor([1, 2, 3]); print(t.sum())"
    elif package_name == "rasterio":
        # Just import - rasterio needs a file to test fully
        return "import rasterio; print(rasterio.__version__)"
    elif package_name == "segment-anything":
        return "import segment_anything; print('ok')"
    elif package_name == "torchvision":
        return "import torchvision; print(torchvision.__version__)"
    else:
        import_name = package_name.replace("-", "_")
        return f"import {import_name}"


def verify_venv(
    venv_dir: str = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not venv_exists(venv_dir):
        return False, "Virtual environment not found"

    python_path = get_venv_python_path(venv_dir)
    env = _get_clean_env_for_venv()
    subprocess_kwargs = _get_subprocess_kwargs()

    total_packages = len(REQUIRED_PACKAGES)
    for i, (package_name, _) in enumerate(REQUIRED_PACKAGES):
        if progress_callback:
            # Report progress for each package (0-100% within verification phase)
            percent = int((i / total_packages) * 100)
            progress_callback(percent, f"Verifying {package_name}... ({i + 1}/{total_packages})")

        # Get functional test code, not just import
        verify_code = _get_verification_code(package_name)
        cmd = [python_path, "-c", verify_code]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                **subprocess_kwargs
            )

            if result.returncode != 0:
                error_detail = result.stderr[:300] if result.stderr else result.stdout[:300]
                _log(
                    f"Package {package_name} verification failed: {error_detail}",
                    Qgis.Warning
                )
                return False, f"Package {package_name} is broken: {error_detail[:100]}"

        except Exception as e:
            _log(f"Failed to verify {package_name}: {str(e)}", Qgis.Warning)
            return False, f"Verification error: {package_name}"

    if progress_callback:
        progress_callback(100, "Verification complete")

    _log("✓ Virtual environment verified successfully", Qgis.Success)
    return True, "Virtual environment ready"


def cleanup_old_libs() -> bool:
    if not os.path.exists(LIBS_DIR):
        return False

    _log("Detected old 'libs/' installation. Cleaning up...", Qgis.Info)

    try:
        shutil.rmtree(LIBS_DIR)
        _log("Old libs/ directory removed successfully", Qgis.Success)
        return True
    except Exception as e:
        _log(f"Failed to remove libs/: {e}. Please delete manually.", Qgis.Warning)
        return False


def create_venv_and_install(
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    cuda_enabled: bool = False
) -> Tuple[bool, str]:
    """
    Complete installation: download Python standalone + create venv + install packages.

    Progress breakdown:
    - 0-10%: Download Python standalone (~50MB)
    - 10-15%: Create virtual environment
    - 15-95%: Install packages (~800MB, or ~2.5GB with CUDA)
    - 95-100%: Verify installation
    """
    from .python_manager import (
        standalone_python_exists,
        download_python_standalone,
        get_python_full_version
    )

    # Log system info for debugging
    _log_system_info()

    # Check for Rosetta emulation on macOS (warning only, don't block)
    rosetta_warning = _check_rosetta_warning()
    if rosetta_warning:
        _log(rosetta_warning, Qgis.Warning)

    # Clean up old venv directories from previous Python versions
    removed_venvs = cleanup_old_venv_directories()
    if removed_venvs:
        _log(f"Removed {len(removed_venvs)} old venv directories", Qgis.Info)

    cleanup_old_libs()

    # Step 1: Download Python standalone if needed
    if not standalone_python_exists():
        python_version = get_python_full_version()
        _log(f"Downloading Python {python_version} standalone...", Qgis.Info)

        def python_progress(percent, msg):
            # Map 0-100 to 0-10
            if progress_callback:
                progress_callback(int(percent * 0.10), msg)

        success, msg = download_python_standalone(
            progress_callback=python_progress,
            cancel_check=cancel_check
        )

        if not success:
            # On Windows, try KADAS Python fallback before giving up
            if sys.platform == "win32":
                kadas_python = _get_kadas_python()
                if kadas_python:
                    _log(
                        "Standalone Python download failed, "
                        "falling back to KADAS Python: {}".format(msg),
                        Qgis.Warning
                    )
                    if progress_callback:
                        progress_callback(10, "Using KADAS Python (fallback)...")
                else:
                    return False, f"Failed to download Python: {msg}"
            else:
                return False, f"Failed to download Python: {msg}"

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("Python standalone already installed", Qgis.Info)
        if progress_callback:
            progress_callback(10, "Python standalone ready")

    # Step 2: Create virtual environment if needed
    if venv_exists():
        _log("Virtual environment already exists", Qgis.Info)
        if progress_callback:
            progress_callback(15, "Virtual environment ready")
    else:
        def venv_progress(percent, msg):
            # Map 10-20 to 10-15
            if progress_callback:
                progress_callback(10 + int(percent * 0.05), msg)

        success, msg = create_venv(progress_callback=venv_progress)
        if not success:
            return False, msg

        if cancel_check and cancel_check():
            return False, "Installation cancelled"

    # Step 3: Install dependencies
    def deps_progress(percent, msg):
        # Map 20-100 to 15-95
        if progress_callback:
            mapped = 15 + int((percent - 20) * 0.80 / 0.80)
            progress_callback(min(mapped, 95), msg)

    success, msg = install_dependencies(
        progress_callback=deps_progress,
        cancel_check=cancel_check,
        cuda_enabled=cuda_enabled
    )

    if not success:
        # Check if it's a Group Policy error
        if "GROUP_POLICY_ERROR" in msg:
            _log("Detected Group Policy blocking venv, recreating with virtualenv...", Qgis.Warning)
            if progress_callback:
                progress_callback(15, "Group Policy detected, switching to virtualenv...")
            
            # Remove the old venv
            import shutil
            venv_dir = get_venv_dir()
            if os.path.exists(venv_dir):
                _log(f"Removing old venv: {venv_dir}", Qgis.Info)
                try:
                    shutil.rmtree(venv_dir)
                except Exception as e:
                    _log(f"Failed to remove old venv: {e}", Qgis.Warning)
            
            # Get system Python
            system_python = _get_system_python()
            
            # Create venv with virtualenv
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            def venv_progress_retry(percent, msg):
                if progress_callback:
                    progress_callback(10 + int(percent * 0.05), msg)
            
            success_venv, msg_venv = _create_venv_with_virtualenv(
                venv_dir, system_python, env, venv_progress_retry
            )
            
            if not success_venv:
                return False, f"Failed to create venv with virtualenv: {msg_venv}"
            
            if cancel_check and cancel_check():
                return False, "Installation cancelled"
            
            # Retry installation with the new venv
            success, msg = install_dependencies(
                progress_callback=deps_progress,
                cancel_check=cancel_check,
                cuda_enabled=cuda_enabled
            )
            
            if not success:
                return False, msg
        else:
            return False, msg

    # Step 4: Verify installation (95-100%)
    def verify_progress(percent: int, msg: str):
        """Map verification progress (0-100%) to overall progress (95-99%)."""
        if progress_callback:
            # Map 0-100 to 95-99 (leave 100% for final success message)
            mapped = 95 + int(percent * 0.04)
            progress_callback(min(mapped, 99), msg)

    is_valid, verify_msg = verify_venv(progress_callback=verify_progress)
    if not is_valid:
        return False, f"Verification failed: {verify_msg}"

    if progress_callback:
        progress_callback(100, "✓ All dependencies installed")

    return True, "Virtual environment ready"


def get_venv_status() -> Tuple[bool, str]:
    """Get the status of the complete installation (Python standalone + venv)."""
    from .python_manager import standalone_python_exists, get_python_full_version

    # Check for old libs/ installation
    if os.path.exists(LIBS_DIR):
        return False, "Old installation detected. Migration required."

    # Check Python standalone
    if not standalone_python_exists():
        return False, "Dependencies not installed"

    # Check venv
    if not venv_exists():
        return False, "Virtual environment not configured"

    # Verify packages
    is_valid, msg = verify_venv()
    if is_valid:
        python_version = get_python_full_version()
        return True, f"Ready (Python {python_version})"
    else:
        return False, f"Virtual environment incomplete: {msg}"


def remove_venv(venv_dir: str = None) -> Tuple[bool, str]:
    if venv_dir is None:
        venv_dir = VENV_DIR

    if not os.path.exists(venv_dir):
        return True, "Virtual environment does not exist"

    try:
        shutil.rmtree(venv_dir)
        _log(f"Removed virtual environment: {venv_dir}", Qgis.Success)
        return True, "Virtual environment removed"
    except Exception as e:
        _log(f"Failed to remove venv: {e}", Qgis.Warning)
        return False, f"Failed to remove venv: {str(e)[:200]}"


def _remove_broken_symlinks(directory: str) -> int:
    """
    Recursively find and remove broken symlinks in a directory.
    Returns the number of broken symlinks removed.
    """
    removed_count = 0
    try:
        for root, dirs, files in os.walk(directory, topdown=False):
            # Check all entries (files and dirs can be symlinks)
            for name in files + dirs:
                path = os.path.join(root, name)
                # Check if it's a symlink (broken or not)
                if os.path.islink(path):
                    # Check if the symlink target exists
                    if not os.path.exists(path):
                        # Broken symlink - remove it
                        try:
                            os.unlink(path)
                            removed_count += 1
                        except Exception:
                            pass
    except Exception:
        pass
    return removed_count


def _remove_all_symlinks(directory: str) -> int:
    """
    Recursively find and remove ALL symlinks in a directory.
    This is more aggressive but ensures Qt can delete the directory.
    Returns the number of symlinks removed.
    """
    removed_count = 0
    try:
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files + dirs:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    try:
                        os.unlink(path)
                        removed_count += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return removed_count


def prepare_for_uninstall() -> bool:
    """
    Prepare plugin directory for uninstallation.

    On macOS, two issues prevent Qt's QDir.removeRecursively() from working:
    1. Extended attributes like 'com.apple.provenance'
    2. Broken symlinks (Qt cannot delete them)

    This function:
    - Removes extended attributes from all files
    - Removes all symlinks (broken or not) that Qt can't handle

    Returns True if cleanup was performed, False otherwise.
    """
    if sys.platform != "darwin":
        return False

    # Directories that may have problematic files
    dirs_to_clean = [
        VENV_DIR,
        os.path.join(SRC_DIR, "python_standalone"),
        os.path.join(SRC_DIR, "checkpoints"),
    ]

    cleaned = False
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            # Step 1: Remove all symlinks (Qt can't handle them properly)
            symlinks_removed = _remove_all_symlinks(dir_path)
            if symlinks_removed > 0:
                _log(f"Removed {symlinks_removed} symlinks from: {dir_path}", Qgis.Info)
                cleaned = True

            # Step 2: Remove extended attributes
            try:
                result = subprocess.run(
                    ["xattr", "-r", "-c", dir_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    _log(f"Cleaned extended attributes from: {dir_path}", Qgis.Info)
                    cleaned = True
            except subprocess.TimeoutExpired:
                _log(f"xattr cleanup timed out for: {dir_path}", Qgis.Warning)
            except FileNotFoundError:
                _log("xattr command not found", Qgis.Warning)
            except Exception as e:
                _log(f"Failed to clean extended attributes: {e}", Qgis.Warning)

    return cleaned
