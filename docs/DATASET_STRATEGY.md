# Dataset Strategy

## Best first dataset

Use USGS 3DEP / 1m DEMs for high-resolution U.S. training patches.

Why:

- bare-earth elevation
- 1m resolution lines up with Minecraft block scale
- lots of mountainous terrain available
- good for DEM super-resolution/detail training

## Secondary sources

- OpenTopography API/global datasets for wider terrain diversity
- NASADEM / Copernicus 30m for macro terrain
- Sentinel-2 later if texture/biome conditioning becomes useful

## Initial regions

Start with diverse U.S. 1m terrain:

- Colorado Rockies
- Sierra Nevada
- Cascades
- Appalachians
- Utah badlands/canyons
- volcanic terrain
- glacial valleys
- foothills/rolling hills

## Split rule

Validation must be by source region/file, not by random tile. Adjacent train/val tiles leak the answer.

## Sourcing

How get? TBD.
