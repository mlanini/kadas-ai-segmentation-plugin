# KADAS AI Segmentation Plugin

[![KADAS](https://img.shields.io/badge/KADAS-Albireo_2-blue?style=flat-square)](https://github.com/kadas-albireo/kadas-albireo2) [![Version](https://img.shields.io/badge/version-0.1.0-green?style=flat-square)]() [![License](https://img.shields.io/badge/license-GPLv2-orange?style=flat-square)](LICENSE)

**Segment anything in your geospatial rasters using AI**

AI-powered object segmentation for raster imagery in KADAS Albireo 2. Uses Meta's Segment Anything Model (SAM2) for intelligent detection and vectorization of objects in satellite/aerial imagery.

---

## Features

- **AI-Powered**: Uses SAM2 for state-of-the-art object segmentation
- **KADAS Native**: Integrated in AI ribbon tab with KADAS optimizations
- **Auto-Setup**: Automatic Python environment and dependency management
- **Interactive**: Click-based positive/negative point segmentation
- **Export**: Save polygons to GeoPackage or Shapefile

## Quick Start

1. **Install**: KADAS → Plugins → Install from ZIP → Select `ai-segmentation-x.x.x.zip`
2. **Enable**: Plugins → Manage and Install Plugins → Enable "Imagery Segmentation"
3. **Open**: AI tab → AI Segmentation
4. **Setup** (first time): Click "Install Dependencies" (~800MB download)
5. **Load Imagery**: Add raster layer to KADAS
6. **Encode**: Select layer → "Encode Layer" button
7. **Segment**: 
   - Left-click on objects (positive points)
   - Ctrl+Click to exclude areas (negative points)
8. **Export**: "Export Layer" button → GeoPackage/Shapefile

## Requirements

- KADAS Albireo 2.x
- ~3GB disk space (Python environment + models)
- Internet connection for initial setup

## Documentation

- [Release Notes](RELEASE_NOTES.md) - Version history and changes
- [GitHub Issues](https://github.com/mlanini/kadas-ai-segmentation-plugin/issues) - Bug reports and features

**In-Plugin Help:**
- Menu → Help (Online) - Opens this README
- Menu → Open Log File - View `~/.kadas/ai_segmentation.log`

## Credits

- **Based on**: [QGIS Plugin](https://github.com/Terradue/qgis-plugin-segment-anything) by TerraLab GmbH
- **SAM2**: Meta AI Research
- **KADAS Adaptation**: Michael Lanini
- **License**: GPLv2

## Support

For issues or questions: [GitHub Issues](https://github.com/mlanini/kadas-ai-segmentation-plugin/issues)

