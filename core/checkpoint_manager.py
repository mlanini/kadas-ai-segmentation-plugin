import os
import hashlib
import threading
from typing import Tuple, Optional, Callable

from qgis.core import QgsMessageLog, Qgis
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from .model_config import (
    CHECKPOINT_URL, CHECKPOINT_FILENAME, CHECKPOINT_SHA256,
    USE_SAM2,
)


def _get_cache_base_dir() -> str:
    """Get the appropriate cache directory for KADAS.
    
    Checks AI_SEGMENTATION_CACHE_DIR environment variable first (for corporate environments),
    then falls back to platform-specific defaults.

    Returns:
        - Environment variable AI_SEGMENTATION_CACHE_DIR if set
        - KADAS: ~/.kadas_ai_segmentation (Linux/Mac) or %APPDATA%/kadas_ai_segmentation (Windows)
    """
    import sys

    # Check environment variable first (for corporate/IT-managed installations)
    custom_cache = os.environ.get("AI_SEGMENTATION_CACHE_DIR")
    if custom_cache:
        os.makedirs(custom_cache, exist_ok=True)
        return custom_cache

    if sys.platform == "win32":
        # Windows: Use %APPDATA%/kadas_ai_segmentation
        appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
        base_dir = os.path.join(appdata, "kadas_ai_segmentation")
    else:
        # Linux/Mac: Use hidden directory in home
        base_dir = os.path.expanduser("~/.kadas_ai_segmentation")

    os.makedirs(base_dir, exist_ok=True)
    return base_dir


CACHE_DIR = _get_cache_base_dir()
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

SAM_CHECKPOINT_URL = CHECKPOINT_URL
SAM_CHECKPOINT_FILENAME = CHECKPOINT_FILENAME
SAM_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
OLD_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"  # noqa: S105  # pragma: allowlist secret


def get_checkpoints_dir() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return CHECKPOINTS_DIR


def get_features_dir() -> str:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    return FEATURES_DIR


def get_raster_features_dir(raster_path: str) -> str:
    import re

    # Get the raster filename without extension
    raster_filename = os.path.splitext(os.path.basename(raster_path))[0]

    # Sanitize: keep only alphanumeric, underscore, hyphen (replace others with underscore)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raster_filename)
    # Limit length to avoid too long folder names
    sanitized_name = sanitized_name[:40]
    # Remove trailing underscores
    sanitized_name = sanitized_name.rstrip('_')

    # Add short hash suffix for uniqueness (based on full path)
    # MD5 is used here only for generating a short identifier, not for security
    raster_hash = hashlib.md5(raster_path.encode(), usedforsecurity=False).hexdigest()[:8]

    # Combine: rastername_abc12345
    folder_name = f"{sanitized_name}_{raster_hash}"

    features_path = os.path.join(FEATURES_DIR, folder_name)
    os.makedirs(features_path, exist_ok=True)
    return features_path


def get_checkpoint_path() -> str:
    return os.path.join(get_checkpoints_dir(), SAM_CHECKPOINT_FILENAME)


def checkpoint_exists() -> bool:
    return os.path.exists(get_checkpoint_path())


def verify_checkpoint_hash(filepath: str) -> bool:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == SAM_CHECKPOINT_SHA256


def download_checkpoint(
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    """
    Download SAM checkpoint using KADAS network manager with progress reporting.
    
    Implements exponential backoff retry strategy (5, 10, 20, 40, 80s delays)
    for network resilience in corporate environments (v0.6.5).

    Uses QNetworkAccessManager with a local event loop to provide real-time
    download progress updates instead of blocking without feedback.
    
    Note: Proxy settings are automatically applied by proxy_handler.initialize_proxy()
    called in plugin initGui(). QgsNetworkAccessManager respects KADAS proxy settings.
    """
    # Exponential backoff configuration (v0.6.5)
    MAX_RETRIES = 5
    BACKOFF_DELAYS = [5, 10, 20, 40, 80]  # seconds
    
    last_error = ""
    
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            delay = BACKOFF_DELAYS[attempt - 1]
            QgsMessageLog.logMessage(
                f"Download failed (attempt {attempt}/{MAX_RETRIES}), "
                f"retrying in {delay}s...",
                "AI Segmentation",
                level=Qgis.Warning
            )
            if progress_callback:
                progress_callback(0, f"Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            import time
            time.sleep(delay)
        
        success, message = _download_checkpoint_attempt(progress_callback)
        
        if success:
            return True, message
        
        last_error = message
        QgsMessageLog.logMessage(
            f"Download attempt {attempt + 1}/{MAX_RETRIES} failed: {message}",
            "AI Segmentation",
            level=Qgis.Warning
        )
    
    # All retries exhausted - provide firewall guidance
    firewall_hint = (
        "\n\nDownload failed after all retries. "
        "This may indicate firewall or network restrictions.\n\n"
        "Please ask your IT department to allow access to:\n"
        "  - download.pytorch.org\n"
        "  - githubusercontent.com\n\n"
        "Or manually download the checkpoint file and place it at:\n"
        f"  {get_checkpoint_path()}"
    )
    
    QgsMessageLog.logMessage(
        f"All download attempts failed. Last error: {last_error}{firewall_hint}",
        "AI Segmentation",
        level=Qgis.Critical
    )
    
    return False, f"{last_error}{firewall_hint}"


def _download_checkpoint_attempt(
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str]:
    """
    Single download attempt (without retry logic).
    Internal function called by download_checkpoint() with exponential backoff.
    """
    from qgis.PyQt.QtCore import QEventLoop, QTimer
    from qgis.PyQt.QtNetwork import QNetworkReply
    from qgis.core import QgsNetworkAccessManager

    checkpoint_path = get_checkpoint_path()
    temp_path = checkpoint_path + ".tmp"

    if checkpoint_exists():
        QgsMessageLog.logMessage(
            "Checkpoint already exists, verifying...",
            "AI Segmentation",
            level=Qgis.Info
        )
        if verify_checkpoint_hash(checkpoint_path):
            return True, "Checkpoint verified"
        else:
            QgsMessageLog.logMessage(
                "Checkpoint hash mismatch, re-downloading...",
                "AI Segmentation",
                level=Qgis.Warning
            )
            os.remove(checkpoint_path)

    if progress_callback:
        progress_callback(0, "Connecting to download server...")

    # State container for progress tracking
    download_state = {
        'bytes_received': 0,
        'bytes_total': 0,
        'error': None,
        'data': bytearray()
    }

    def on_download_progress(received: int, total: int):
        """Handle download progress updates."""
        download_state['bytes_received'] = received
        download_state['bytes_total'] = total

        if total > 0 and progress_callback:
            # Calculate percentage (reserve 5% for verification)
            percent = int((received / total) * 90) + 5
            mb_received = received / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            progress_callback(
                min(percent, 95),
                f"Downloading: {mb_received:.1f} / {mb_total:.1f} MB"
            )
        elif progress_callback and received > 0:
            # Unknown total size, show bytes downloaded
            mb_received = received / (1024 * 1024)
            progress_callback(50, f"Downloading: {mb_received:.1f} MB...")

    def on_ready_read():
        """Accumulate downloaded data chunks."""
        data = reply.readAll()
        download_state['data'].extend(data.data())

    def on_error(error_code):
        """Handle download errors."""
        download_state['error'] = reply.errorString()

    try:
        # Use KADAS network manager for proxy-aware downloads
        manager = QgsNetworkAccessManager.instance()
        qurl = QUrl(SAM_CHECKPOINT_URL)
        request = QNetworkRequest(qurl)

        # Start the download
        reply = manager.get(request)

        # Connect progress signals
        reply.downloadProgress.connect(on_download_progress)
        reply.readyRead.connect(on_ready_read)
        reply.errorOccurred.connect(on_error)

        # Create event loop to wait for download completion
        loop = QEventLoop()
        reply.finished.connect(loop.quit)

        # Add timeout (20 minutes for ~375MB file on slow connections)
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        timeout.start(1200000)  # 20 minutes

        if progress_callback:
            progress_callback(5, "Download started...")

        # Run the event loop until download completes
        loop.exec_()

        # Check for timeout
        if timeout.isActive():
            timeout.stop()
        else:
            reply.abort()
            return False, "Download timed out after 20 minutes"

        # Check for errors
        if reply.error() != QNetworkReply.NoError:
            error_msg = download_state['error'] or reply.errorString()
            QgsMessageLog.logMessage(
                f"Checkpoint download failed: {error_msg}",
                "AI Segmentation",
                level=Qgis.Warning
            )
            reply.deleteLater()
            return False, f"Download failed: {error_msg}"

        # Read any remaining data
        remaining = reply.readAll()
        if remaining:
            download_state['data'].extend(remaining.data())

        content = bytes(download_state['data'])
        reply.deleteLater()

        if len(content) == 0:
            return False, "Download failed: empty response"

        if progress_callback:
            mb_total = len(content) / (1024 * 1024)
            progress_callback(92, f"Downloaded {mb_total:.1f} MB, saving...")

        # Write content to temp file
        with open(temp_path, 'wb') as f:
            f.write(content)

        # Check for 0-byte corrupted download (v0.6.5)
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            os.remove(temp_path)
            QgsMessageLog.logMessage(
                "Downloaded checkpoint is 0 bytes (corrupt)",
                "AI Segmentation",
                level=Qgis.Critical
            )
            return False, "Download corrupted (0 bytes)"

        if progress_callback:
            progress_callback(95, "Verifying download...")

        if not verify_checkpoint_hash(temp_path):
            os.remove(temp_path)
            return False, "Download verification failed - hash mismatch"

        os.replace(temp_path, checkpoint_path)

        if progress_callback:
            progress_callback(100, "Checkpoint downloaded successfully!")

        QgsMessageLog.logMessage(
            f"Checkpoint downloaded to: {checkpoint_path}",
            "AI Segmentation",
            level=Qgis.Success
        )
        return True, "Checkpoint downloaded and verified"

    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        QgsMessageLog.logMessage(
            f"Checkpoint download failed: {str(e)}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False, f"Download failed: {str(e)}"


def has_features_for_raster(raster_path: str) -> bool:
    features_dir = get_raster_features_dir(raster_path)
    csv_path = os.path.join(features_dir, os.path.basename(features_dir) + ".csv")
    if not os.path.exists(csv_path):
        return False
    tif_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
    return len(tif_files) > 0


def clear_features_for_raster(raster_path: str) -> bool:
    features_dir = get_raster_features_dir(raster_path)
    if os.path.exists(features_dir):
        import shutil
        shutil.rmtree(features_dir)
        os.makedirs(features_dir, exist_ok=True)
        return True
    return False


def cleanup_legacy_sam1_data():
    """Remove legacy SAM1 data: old checkpoint and features cache.

    Called once at plugin startup. Errors are logged silently
    to avoid disturbing the user.

    On Python 3.9 (USE_SAM2=False), the SAM1 checkpoint is still needed,
    so only features cache is cleaned up.
    """
    import shutil

    # Remove old SAM1 checkpoint only when using SAM2 (Python 3.10+)
    if USE_SAM2:
        old_checkpoint = os.path.join(CHECKPOINTS_DIR, OLD_CHECKPOINT_FILENAME)
        if os.path.exists(old_checkpoint):
            try:
                os.remove(old_checkpoint)
                QgsMessageLog.logMessage(
                    "Removed old SAM1 checkpoint: {}".format(
                        OLD_CHECKPOINT_FILENAME),
                    "AI Segmentation", level=Qgis.Info)
            except OSError as e:
                QgsMessageLog.logMessage(
                    "Could not remove old checkpoint: {}".format(e),
                    "AI Segmentation", level=Qgis.Warning)

    # Remove old features cache (SAM2 does on-demand encoding, no cache needed)
    if os.path.exists(FEATURES_DIR):
        try:
            shutil.rmtree(FEATURES_DIR)
            QgsMessageLog.logMessage(
                "Removed legacy features cache",
                "AI Segmentation", level=Qgis.Info)
        except OSError as e:
            QgsMessageLog.logMessage(
                "Could not remove features cache: {}".format(e),
                "AI Segmentation", level=Qgis.Warning)
