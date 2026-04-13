# KADAS AI Segmentation Plugin

[![KADAS](https://img.shields.io/badge/KADAS-Albireo_2-blue?style=flat-square)](https://github.com/kadas-albireo/kadas-albireo2) [![Version](https://img.shields.io/badge/version-0.9.5-green?style=flat-square)]() [![License](https://img.shields.io/badge/license-GPLv2-orange?style=flat-square)](LICENSE)

**Segment anything in your geospatial rasters using AI**

AI-powered object segmentation for raster imagery in KADAS Albireo 2. Uses Meta's Segment Anything Model (SAM2, with SAM1 fallback) for intelligent detection and vectorization of objects in satellite/aerial imagery.

> Based on [QGIS AI-Segmentation](https://github.com/TerraLabAI/QGIS_AI-Segmentation) v0.9.5 by TerraLab GmbH.
> Full tutorial: https://terra-lab.ai/ai-segmentation

---

## Features

### AI & Segmentation
- **SAM2 + SAM1 fallback** — SAM2 on Python 3.10+, automatic SAM1 fallback on older Python
- **On-demand crop encoding** — Click anywhere to encode a local crop; no full-raster pre-encoding
- **Frozen crop sessions** — Freeze the current crop and continue segmenting in other areas
- **Interactive points** — Left-click (positive) / Ctrl+Click (negative) to refine boundaries
- **Per-point undo** — Ctrl+Z to undo the last point, with 30-level mask history

### Layer Support
- **Local rasters** — GeoTIFF, JPG, PNG and other GDAL-supported formats
- **Online layers** — WMS, XYZ tiles, WMTS, WCS, ArcGIS MapServer/ImageServer
- **Non-georeferenced images** — Works on imagery without CRS metadata
- **CRS transforms** — Automatic reprojection between canvas CRS and raster CRS

### Refinement & Export
- **Polygon refinement** — Expand, contract, simplify, smooth, fill holes
- **GeoPackage export** — Layer groups with configurable naming
- **Shapefile export** — Traditional vector export
- **Disjoint region warnings** — Alerts when mask contains separate polygons

### User Experience
- **Keyboard shortcuts** — Space (pan), Ctrl+Z (undo), S (save), Enter (confirm), Esc (cancel)
- **Plugin update checks** — Notifications when a new version is available
- **Automatic setup** — Python environment, uv package manager, and dependencies install automatically
- **Multilingual** — English, French, German, Spanish, Portuguese, Italian (+ Swiss locales)

### KADAS-Specific
- **Ribbon tab integration** — AI tab → AI Segmentation
- **Proxy/VPN support** — Enterprise network compatibility
- **Dedicated cache** — Isolated environment in `~/.kadas_ai_segmentation`

### Diagnostics
- **Error diagnostics** — SSL, DLL, antivirus conflict detection
- **Bug report dialog** — Automatic log collection with path anonymization
- **PyTorch DLL guidance** — VC++ Redistributables detection and advice

---

## Quick Start

1. **Install**: KADAS → Plugins → Install from ZIP → Select `ai-segmentation-0.9.5.zip`
2. **Enable**: Plugins → Manage and Install Plugins → Enable "Imagery Segmentation"
3. **Open**: AI tab → AI Segmentation
4. **Setup** (first time): Click "Install Dependencies" (automatic download ~800MB)
5. **Load Imagery**: Add a raster or online layer to KADAS
6. **Segment**:
   - Left-click on the map → a crop is automatically encoded around the click
   - Continue clicking to add positive points on the target object
   - Ctrl+Click to add negative points (areas to exclude)
   - Ctrl+Z to undo the last point
7. **Refine**: Use expand/contract/simplify/smooth controls to adjust the polygon
8. **Save**: Press S or click "Save Polygon" to add to the result layer
9. **Export**: "Export Layer" button → GeoPackage or Shapefile

---

## Requirements

- KADAS Albireo 2.x (QGIS 3.22+)
- Python 3.10+ recommended (required for SAM2; SAM1 fallback on 3.9)
- ~3GB disk space (Python environment + SAM model checkpoints)
- Internet connection for initial setup and online layers

---

## Documentation

- [Release Notes](RELEASE_NOTES.md) — Version history and changes
- [GitHub Issues](https://github.com/mlanini/kadas-ai-segmentation-plugin/issues) — Bug reports and feature requests
- [TerraLab Tutorial](https://terra-lab.ai/ai-segmentation) — Full usage guide (upstream)

**In-Plugin Help:**
- Menu → Help (Online) — Opens the TerraLab tutorial
- Menu → Open Log File — View `~/.kadas_ai_segmentation/ai_segmentation.log`
- Menu → Report Bug — Generates a diagnostic report with anonymized logs

---

## Credits

- **Based on**: [QGIS AI-Segmentation](https://github.com/TerraLabAI/QGIS_AI-Segmentation) v0.9.5 by TerraLab.AI
- **SAM2**: Meta AI Research
- **KADAS Adaptation**: Michael Lanini
- **License**: GPLv2

## Support

For issues or questions: [GitHub Issues](https://github.com/mlanini/kadas-ai-segmentation-plugin/issues)

