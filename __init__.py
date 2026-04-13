"""KADAS AI Segmentation plugin entry point."""


def classFactory(iface):
    """Entry point for KADAS Albireo 2 plugin loader.
    
    The loader calls this function and expects an object exposing 
    the plugin interface (initGui/unload, etc.).
    
    Args:
        iface: KADAS KadasPluginInterface instance
        
    Returns:
        AISegmentationPlugin instance
    """
    # Import late to avoid issues during plugin loading
    from .ai_segmentation_plugin import AISegmentationPlugin
    
    # Cleanup old installations on first load
    try:
        from .core.venv_manager import cleanup_old_libs
        cleanup_old_libs()
    except Exception:
        pass  # Don't fail plugin loading due to cleanup errors

    try:
        from .core.checkpoint_manager import cleanup_legacy_sam1_data
        cleanup_legacy_sam1_data()
    except Exception:
        pass  # Don't fail plugin loading due to cleanup errors
    
    return AISegmentationPlugin(iface)
