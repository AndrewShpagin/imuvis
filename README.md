# IMU/vision feedback explorer

Install Python 3 with Tk support, then install the dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

For optional in-window CSV drag-and-drop, install `tkinterdnd2`:

```powershell
py -3 -m pip install tkinterdnd2
```

Start the interactive explorer with:

```powershell
py -3 imuviso.py
```

When `height.csv` is present in the working directory or beside the script, it
loads automatically. To open another log, pass its path or use **Browse**:

```powershell
py -3 imuviso.py C:\path\to\flight.csv
```

On Windows, you can also double-click `imuviso_ui.cmd` or drop one CSV onto
that command file. The open window also accepts a dropped CSV when the optional
`tkinterdnd2` package is installed.

The controls update the preview without writing output files:

All legends stay at the top left. Plot headings and the status line show the
selected window's average height (`mheight`) in meters.

- **Scale (%)** changes the initial vision scale.
- **Timeline (start tick)** above Scale moves through the CSV in whole ticks,
  stays synchronized with Start tick during playback and Reset, and stops at
  the last start tick that fits the selected window. Dragging pauses playback.
- **Start tick** selects the first CSV data row.
- **Play/Pause** advances the start tick by one after each preview finishes;
  playback therefore runs as fast as calculation and drawing allow.
- **Reset** pauses playback and returns to the start tick used when the window
  opened (zero by default).
- **Tick count** changes the window length.
- **Position feedback** shows only iteration 0 and the configured final
  iteration (iteration 4 by default). Each row compares the re-integrated
  trajectory with the fixed initial scaled vision, defined as
  `scale * original_vision`. It shows scaled-vision RMSE
  `sqrt(mean((integration - scaled_vision)^2))` and ordinary average distance
  `mean(abs(integration - scaled_vision))` for the current window.
  For display only, this tab removes a least-squares constant-offset and
  constant-speed line. Scaled vision and each displayed integration share the
  same line, so their sample-by-sample difference is unchanged. The feedback
  coordinate uses the line fitted to its iteration-0 coordinate, and raw
  vision is independently detrended as shape-only context. At 100% scale it
  exactly overlaps the scaled-vision curve, so a dotted marker trace keeps the
  raw curve identifiable without changing either value. All title metrics
  still use the untouched absolute arrays.
  Its figure heading shows the final iteration's average 2D
  integration-to-initial-scaled-vision distance as the scale criterion. Raw
  vision remains visible as context, while the updated feedback coordinate is
  a separate curve. The final row also overlays the initial re-biased
  acceleration integration as a dashed purple line for direct comparison.
- **Realtime endpoints** adds the last sample from each completed window for
  both initial scaled vision and the final re-integrated trajectory. It retains
  at most Tick count samples. Reset, backward movement, a jump over 10 ticks,
  or a change of CSV, Scale, or Tick count clears the accumulated history.
  Each plot title reports endpoint RMSE `sqrt(mean(delta^2))`, average absolute
  difference `mean(abs(delta))`, and mean signed error over the retained
  samples. The figure heading reports the average 2D endpoint distance
  `mean(sqrt(dx^2 + dy^2))` as a single scale criterion. This tab remains in
  absolute coordinates and does not apply the display detrending.

These metrics always hold `scale * raw vision` fixed as their reference; they
do not use the iteratively updated feedback coordinate. The scale with the
smallest distance is therefore the candidate scale for the true-height
experiment. The window still changes acceleration and fits motion and bias
using that same vision window, so independent position truth or a held-out
window remains a stronger validation. In JSON reports, `rmse_m` uses initial
scaled vision; `fit_rmse_m` is reserved for the internal fit against the
iteratively updated coordinate. Explicit `original_vision_*` fields remain as
raw-vision diagnostics.

The feedback uses the NED sign convention
`DX = gain * (integrated - coordinate)`, followed by
`coordinate += DX` and `acceleration += DX * G / height`. Thus a positive
coordinate correction always produces a positive acceleration correction.
The UI warns when `|DX/height|` exceeds 0.1, where the small-angle
approximation deserves inspection, or when initial-scaled-vision RMSE does
not improve on every iteration.

The original batch/export command remains available when all three positional
arguments are supplied:

```powershell
py -3 imuviso.py height.csv 0 500 --no-show
```

Input files are semicolon-separated and require `vis_x`, `vis_y`, `mheight`,
and either `acc.x`/`acc.y` or `acc_x`/`acc_y` columns.
When a positive `G` column is present, its median value is used for the
coordinate-to-acceleration correction. `--gravity` overrides it; otherwise the
fallback is 9.80665 m/s².

Batch mode skips both CSV exports by default to reduce output time. Visual
feedback, saved figures, and the JSON report remain available. Add
`--write-csv` to a batch command when you need the CSV exports. The interactive
preview continues to write no output files.

In the exported `*_true.csv`, `true_PERCENT_physical_acc_x/y` is raw
acceleration plus the geometric `DX*G/height` correction.
`true_PERCENT_acc_x/y` additionally contains the newly fitted constant bias;
this is the acceleration that is integrated for the fitted trajectory.
