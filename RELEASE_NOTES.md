# Release Notes

## Version 0.9.5 (2025-06-25)

### Aligned with upstream QGIS AI-Segmentation v0.9.5

**Architecture Changes:**
- ✅ On-demand crop-based encoding (click-to-encode replaces full-raster pre-encoding)
- ✅ SAM2 support (Python 3.10+) with SAM1 fallback
- ✅ Online layer support (WMS, XYZ, WMTS, WCS, ArcGIS)
- ✅ Fast package installation via uv (Astral)

**New Features:**
- ✅ Refinement controls: expand, contract, simplify, smooth, fill holes
- ✅ Frozen crop sessions for multi-area segmentation
- ✅ Per-point undo with mask state history (30-level cap)
- ✅ Keyboard shortcuts (Space pan, Ctrl+Z, S, Enter, Esc)
- ✅ GeoPackage export with layer groups
- ✅ Plugin update check notifications
- ✅ CRS transforms (canvas CRS vs raster CRS)
- ✅ Non-georeferenced image support
- ✅ Disjoint region detection warnings

**Diagnostics & Error Handling:**
- ✅ Improved error diagnostics (SSL, DLL, antivirus detection)
- ✅ Bug report dialog with log collection and path anonymization
- ✅ PyTorch DLL error detection with VC++ Redistributables guidance

**KADAS-Specific (preserved):**
- ✅ KADAS Albireo 2 ribbon tab integration
- ✅ Proxy/VPN support for enterprise networks
- ✅ KADAS cache directory (~/.kadas_ai_segmentation)
- ✅ KadasPluginInterface.cast(iface) support

## Version 0.1.0 (2026-03-09)

### Initial KADAS Albireo 2 Release

**Major Features:**
- Full KADAS Albireo 2 compatibility
- AI-powered segmentation using SAM2 (Segment Anything Model 2)
- Integrated in KADAS AI ribbon tab
- Interactive point-based segmentation tool
- Automatic Python environment management
- Dependency auto-installation (~800MB)
- Export to GeoPackage and Shapefile formats

**KADAS-Specific Features:**
- Native KADAS plugin interface integration
- Custom dock widget with KADAS styling
- KADAS-optimized map tools
- Ribbon tab integration (AI → AI Segmentation)

**User Interface:**
- Clean, minimal dock widget
- Progress indicators for encoding/prediction
- Interactive map tool with positive/negative points
- Visual feedback with rubber bands
- Status messages in KADAS message bar

**Core Functionality:**
- Layer encoding with SAM2 image embeddings
- Real-time prediction on user clicks
- Multi-point refinement (positive + negative)
- Automatic polygon creation from masks
- Batch export with configurable formats

**Developer Features:**
- Comprehensive logging system (`~/.kadas_ai_segmentation/ai_segmentation.log`)
- Error reporting with detailed traceback
- Menu actions: "Open Log File" and "Help (Online)"
- Automatic cleanup of legacy installations
- Virtual environment isolation

**Technical Details:**
- Flat plugin structure (KADAS-compatible)
- Python 3.12 compatibility
- CUDA/CPU automatic detection
- Checkpoint manager for SAM2 models
- Rasterio/GDAL integration for geospatial data

**Installation:**
- Plugin name: `ai_segmentation` (with underscore)
- Package name: `ai-segmentation-x.x.x.zip` (with hyphen)
- Auto-deploy option for development
- Metadata includes QGIS version compatibility (3.22-4.99)

**Known Limitations:**
- Requires ~3GB disk space (environment + models)
- Initial setup requires internet connection
- Large rasters may require significant processing time
- CUDA support requires NVIDIA GPU with compatible drivers

**Documentation:**
- [README.md](README.md) — Quick start guide
- [LICENSE](LICENSE) — GPLv2 license

**Links:**
- Repository: https://github.com/mlanini/kadas-ai-segmentation-plugin
- Issues: https://github.com/mlanini/kadas-ai-segmentation-plugin/issues
- KADAS: https://github.com/kadas-albireo/kadas-albireo2
- Upstream: https://github.com/TerraLabAI/QGIS_AI-Segmentation
