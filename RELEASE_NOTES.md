# Release Notes

## Version 0.1.0 (2026-03-09)

### 🎉 Initial KADAS Albireo 2 Release

**Major Features:**
- ✅ Full KADAS Albireo 2 compatibility
- ✅ AI-powered segmentation using SAM2 (Segment Anything Model 2)
- ✅ Integrated in KADAS AI ribbon tab
- ✅ Interactive point-based segmentation tool
- ✅ Automatic Python environment management
- ✅ Dependency auto-installation (~800MB)
- ✅ Export to GeoPackage and Shapefile formats

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
- Comprehensive logging system (`~/.kadas/ai_segmentation.log`)
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
- Package name: `ai-segmentation-0.1.0.zip` (with hyphen)
- Auto-deploy option for development
- Metadata includes QGIS version compatibility (3.0-3.99)

**Known Limitations:**
- Requires ~3GB disk space (environment + models)
- Initial setup requires internet connection
- Large rasters may require significant processing time
- CUDA support requires NVIDIA GPU with compatible drivers

**Documentation:**
- [README.md](README.md) - Quick start guide
- [KADAS_TESTING.md](KADAS_TESTING.md) - Complete testing walkthrough
- [LICENSE](LICENSE) - GPLv2 license

**Credits:**
- Based on QGIS plugin by TerraLab GmbH
- SAM2 model by Meta AI Research
- KADAS adaptation by Michael Lanini

**Links:**
- Repository: https://github.com/mlanini/kadas-ai-segmentation-plugin
- Issues: https://github.com/mlanini/kadas-ai-segmentation-plugin/issues
- KADAS: https://github.com/kadas-albireo/kadas-albireo2

---

## Roadmap

**Planned for 0.2.0:**
- [ ] Multiple model size support (tiny, small, large)
- [ ] Batch processing multiple layers
- [ ] Custom model checkpoint loading
- [ ] Performance optimizations for large rasters
- [ ] Additional export formats (GeoJSON, KML)

**Future Considerations:**
- [ ] Automatic object detection mode (no clicks needed)
- [ ] Fine-tuning on custom datasets
- [ ] Integration with other KADAS AI tools
- [ ] Multi-language support (i18n)
- [ ] Plugin settings dialog

---

## Migration from QGIS Plugin

If you were using the original TerraLab QGIS plugin:

**Changes:**
- Plugin renamed: "Segment Anything" → "Imagery Segmentation"
- Flat structure: Files moved from `src/` to root level
- KADAS interface: Uses `KadasPluginInterface` instead of `QgisInterface`
- Ribbon integration: AI tab instead of toolbar
- Import paths: Updated for flat structure

**Compatibility:**
- Virtual environments are isolated (no conflicts)
- Existing models can be reused (same checkpoint paths)
- Export formats unchanged (GeoPackage/Shapefile)

**Migration Steps:**
1. Uninstall old QGIS plugin (if installed)
2. Install KADAS version from ZIP
3. Re-encode layers (old embeddings not compatible)
4. Dependencies will be reinstalled automatically

---

## Changelog Format

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) principles.

**Categories:**
- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes
