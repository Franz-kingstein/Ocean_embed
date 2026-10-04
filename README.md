# OceanEmbed — Interactive Map Demonstration

A self-contained web app that visualises the OceanEmbed model's subsurface
temperature reconstruction over the Bay of Bengal (Jan 2020 pilot).

## Run

No build step. Open directly:

```
xdg-open index.html
```

Or, if you prefer a local server:

```
python3 -m http.server 8777
# then open http://127.0.0.1:8777/index.html
```

## How to use it

- **Map** — shows the predicted temperature field for the selected day.
  Default view is *Surface* (SST); switch to *At depth* and pick a depth.
- **Date** — scrub through January 2020.
- **Field source** — toggle between the *Model* prediction and the *GLORYS12
  Reference* it was trained against, to compare.
- **Click anywhere on the map**, or type a latitude/longitude and press
  *Read profile here*, to plot the temperature–depth profile for that point.
- **Profile chart** — copper line = model, blue line = reference; the bend is
  the thermocline.

## The three numbered steps in the panel

1. Nine satellite surface measurements for one day.
2. A neural network converts them into temperature at depth.
3. The map is that prediction; click a point for its depth profile.

## Data

`data/data.js` holds the model's actual output — 31 daily fields, each
14 depths × 33 × 41 cells — exported by the training pipeline and embedded as
base64. `data/coast.js` is a simplified coastline (Natural Earth) for context.
Nothing is fetched at runtime; the app is fully offline.

## Validity

Pilot study: one month, one basin (82–92°E, 10–18°N). Validation against real
ARGO float profiles on strictly held-out days gives RMSE 0.791 °C,
correlation 0.993. Reference is the GLORYS12 reanalysis. This is a proof of
concept; the full domain and multi-year training is the next phase.
