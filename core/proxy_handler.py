# -*- coding: utf-8 -*-
"""
Proxy and VPN Handler for KADAS AI Segmentation Plugin

Handles proxy detection and configuration based on KADAS/QGIS settings.
Inspired by kadas-vantor-plugin and kadas-stac-plugin patterns.
Uses QgsNetworkAccessManager which automatically respects QGIS proxy settings.
"""

import os
import socket
from typing import Dict, Optional

from qgis.core import QgsSettings, QgsNetworkAccessManager, QgsMessageLog, Qgis
from qgis.PyQt.QtNetwork import QNetworkProxy, QNetworkProxyFactory


# Global proxy configuration
PROXY_CONFIG = {
    'initialized': False,
    'enabled': False,
    'proxy_type': None,
    'host': '',
    'port': 0,
    'user': '',
    'password': '',
    'excludes': '',
    'is_vpn': False
}


def get_qgis_proxy_settings() -> Dict:
    """
    Reads proxy configuration from QGIS Settings.
    
    Returns:
        Dict: Proxy configuration with keys:
            - enabled: Proxy enabled (bool)
            - type: Proxy type string ('HttpProxy', 'Socks5Proxy', etc.)
            - host: Proxy host (str)
            - port: Proxy port (int)
            - user: Username (str, optional)
            - password: Password (str, optional)
            - excludes: NO_PROXY hosts (str, optional)
    """
    settings = QgsSettings()
    
    # Read proxy settings from QGIS
    proxy_enabled = settings.value("proxy/enabled", False, type=bool)
    proxy_type = settings.value("proxy/type", "HttpProxy", type=str)
    proxy_host = settings.value("proxy/host", "", type=str)
    proxy_port = settings.value("proxy/port", 0, type=int)
    proxy_user = settings.value("proxy/user", "", type=str)
    proxy_password = settings.value("proxy/password", "", type=str)
    proxy_excludes = settings.value("proxy/excludes", "", type=str)
    
    return {
        'enabled': proxy_enabled,
        'type': proxy_type,
        'host': proxy_host,
        'port': proxy_port,
        'user': proxy_user,
        'password': proxy_password,
        'excludes': proxy_excludes
    }


def build_proxy_url(settings: Dict) -> Optional[str]:
    """
    Builds proxy URL from QGIS settings.
    
    Args:
        settings: QGIS proxy settings dict
        
    Returns:
        Proxy URL string or None
    """
    if not settings.get('enabled') or not settings.get('host'):
        return None
    
    host = settings['host']
    port = settings['port']
    user = settings.get('user', '')
    password = settings.get('password', '')
    
    # Build URL with or without authentication
    if user and password:
        return f"http://{user}:{password}@{host}:{port}"
    else:
        return f"http://{host}:{port}"


def detect_vpn_connection() -> bool:
    """
    Heuristic VPN detection.
    
    Attempts to detect if a VPN connection is active by checking the default gateway.
    If the gateway is in a private IP range, it's likely a local network.
    
    Returns:
        bool: True if VPN suspected
    """
    try:
        # Get default gateway by checking local hostname IP
        gw = socket.gethostbyname(socket.gethostname())
        
        # If gateway starts with private IP ranges, likely NOT VPN
        if gw.startswith("10.") or gw.startswith("172.") or gw.startswith("192.168."):
            QgsMessageLog.logMessage(
                "Connection appears to be local network (not VPN)",
                "AI Segmentation",
                level=Qgis.Info
            )
            return False
        else:
            QgsMessageLog.logMessage(
                "Connection may be through VPN or public network",
                "AI Segmentation",
                level=Qgis.Info
            )
            return True
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Could not determine VPN status: {e}",
            "AI Segmentation",
            level=Qgis.Warning
        )
        return False


def apply_proxy_settings():
    """
    Applies proxy settings from QGIS to Qt and environment variables.
    
    This function:
    1. Reads proxy settings from QGIS Settings
    2. Applies them to Qt's QNetworkProxy
    3. Sets environment variables for external libraries (pip, etc.)
    4. Detects VPN connections
    """
    global PROXY_CONFIG
    
    settings = get_qgis_proxy_settings()
    
    proxy_vars = (
        "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
        "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"
    )
    
    if not settings['enabled']:
        # Proxy disabled - use system configuration
        QNetworkProxyFactory.setUseSystemConfiguration(True)
        QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.NoProxy))
        
        QgsMessageLog.logMessage(
            "Proxy disabled: using system configuration",
            "AI Segmentation",
            level=Qgis.Info
        )
        
        # Remove all proxy environment variables
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
        
        PROXY_CONFIG.update({
            'initialized': True,
            'enabled': False
        })
        return
    
    # Proxy enabled - apply settings
    host = settings['host']
    port = settings['port']
    user = settings.get('user', '')
    password = settings.get('password', '')
    excludes = settings.get('excludes', '')
    proxy_type = settings['type']
    
    # Map QGIS proxy type to Qt proxy type
    qt_type_map = {
        "HttpProxy": QNetworkProxy.HttpProxy,
        "HttpCachingProxy": QNetworkProxy.HttpCachingProxy,
        "Socks5Proxy": QNetworkProxy.Socks5Proxy,
        "FtpCachingProxy": QNetworkProxy.FtpCachingProxy,
    }
    
    qproxy = QNetworkProxy(
        qt_type_map.get(proxy_type, QNetworkProxy.HttpProxy),
        host, port, user, password
    )
    QNetworkProxy.setApplicationProxy(qproxy)
    QNetworkProxyFactory.setUseSystemConfiguration(False)
    
    QgsMessageLog.logMessage(
        f"Proxy applied: {proxy_type}://{host}:{port} (user: {user})",
        "AI Segmentation",
        level=Qgis.Info
    )
    
    # Set environment variables for external libraries (pip, wget, etc.)
    if host and port:
        scheme = "socks5h" if proxy_type.startswith("Socks5") else "http"
        cred = f"{user}:{password}@" if user else ""
        proxy_url = f"{scheme}://{cred}{host}:{port}"
        
        for var in proxy_vars:
            if var.lower().startswith("no_proxy"):
                continue
            os.environ[var] = proxy_url
        
        if excludes:
            os.environ["NO_PROXY"] = excludes
            os.environ["no_proxy"] = excludes
    else:
        # Invalid host/port - remove environment variables
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
    
    # Detect VPN
    is_vpn = detect_vpn_connection()
    
    # Update global configuration
    PROXY_CONFIG.update({
        'initialized': True,
        'enabled': True,
        'proxy_type': proxy_type,
        'host': host,
        'port': port,
        'user': user,
        'password': password,
        'excludes': excludes,
        'is_vpn': is_vpn
    })


def initialize_proxy():
    """
    Initialize proxy configuration.
    
    Should be called once at plugin startup.
    Safe to call multiple times - only initializes once.
    """
    global PROXY_CONFIG
    
    if PROXY_CONFIG.get('initialized', False):
        QgsMessageLog.logMessage(
            "Proxy already initialized",
            "AI Segmentation",
            level=Qgis.Info
        )
        return
    
    apply_proxy_settings()
    
    QgsMessageLog.logMessage(
        "Proxy initialization complete",
        "AI Segmentation",
        level=Qgis.Info
    )


def get_proxy_config() -> Dict:
    """
    Returns the current proxy configuration.
    
    If proxy not yet initialized, initialize_proxy() will be called.
    
    Returns:
        Dict: Proxy configuration
    """
    global PROXY_CONFIG
    
    if not PROXY_CONFIG.get('initialized', False):
        initialize_proxy()
    
    return PROXY_CONFIG.copy()


def is_proxy_enabled() -> bool:
    """
    Checks if proxy is enabled.
    
    Returns:
        bool: True if proxy enabled
    """
    config = get_proxy_config()
    return config.get('enabled', False)


def is_vpn_detected() -> bool:
    """
    Checks if VPN connection was detected.
    
    Returns:
        bool: True if VPN detected
    """
    config = get_proxy_config()
    return config.get('is_vpn', False)


def get_network_manager():
    """
    Returns QgsNetworkAccessManager instance with proxy applied.
    
    QgsNetworkAccessManager automatically respects QGIS proxy settings,
    so this is mainly for ensuring proxy initialization.
    
    Returns:
        QgsNetworkAccessManager instance
    """
    # Ensure proxy is initialized
    if not PROXY_CONFIG.get('initialized', False):
        initialize_proxy()
    
    return QgsNetworkAccessManager.instance()
