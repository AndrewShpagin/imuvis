"""Fit integrated NED acceleration to vision positions in a flight-log window.

Model: p0 + v0*t + double_integral(acc + bias + G*tilt_correction)
       = scale*vis + tilt_correction*height.
The fitted tilt value is the signed correction to apply, rather than the raw
physical tilt error. Samples use a fixed 0.05 s interval; acceleration is
linear between samples.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DT = 0.05
STANDARD_GRAVITY = 9.80665
SMALL_ANGLE_RATIO_WARNING = 0.1


@dataclass(frozen=True)
class FlightData:
    """Required flight-log columns loaded once for batch or interactive use."""

    path: Path
    acc_x: np.ndarray
    acc_y: np.ndarray
    vis_x: np.ndarray
    vis_y: np.ndarray
    height: np.ndarray
    source_columns: dict
    gravity_m_s2: float | None = None

    @property
    def row_count(self):
        return len(self.height)


def resolve_default_csv(explicit=None):
    """Resolve an explicit CSV or the nearest default height.csv."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    candidates = [Path.cwd() / 'height.csv', Path(__file__).resolve().with_name('height.csv')]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_flight_csv(csv_path):
    """Load the five required columns, retaining invalid cells as NaN for window validation."""
    path = Path(csv_path).expanduser().resolve()
    try:
        source = path.open(newline='', encoding='utf-8-sig')
    except OSError as exc:
        raise ValueError(f'Cannot open CSV {path}: {exc}') from exc

    with source:
        reader = csv.reader(source, delimiter=';')
        try:
            fields = next(reader)
        except StopIteration as exc:
            raise ValueError(f'CSV is empty: {path}') from exc
        fields = [field.strip() for field in fields]
        aliases = {
            'acc_x': ('acc_x', 'acc.x'),
            'acc_y': ('acc_y', 'acc.y'),
            'vis_x': ('vis_x',),
            'vis_y': ('vis_y',),
            'mheight': ('mheight',),
        }
        source_columns = {}
        indexes = []
        missing = []
        for canonical, choices in aliases.items():
            matches = [name for name in choices if name in fields]
            if not matches:
                missing.append('/'.join(choices))
                continue
            selected = matches[0]
            source_columns[canonical] = selected
            indexes.append(fields.index(selected))
        if missing:
            raise ValueError('CSV is missing required columns: ' + ', '.join(missing))
        gravity_index = fields.index('G') if 'G' in fields else None
        if gravity_index is not None:
            source_columns['gravity'] = 'G'

        rows = []
        gravity_values = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            values = []
            for index in indexes:
                try:
                    values.append(float(row[index]))
                except (IndexError, TypeError, ValueError):
                    values.append(float('nan'))
            rows.append(values)
            if gravity_index is not None:
                try:
                    gravity_values.append(float(row[gravity_index]))
                except (IndexError, TypeError, ValueError):
                    gravity_values.append(float('nan'))

    if not rows:
        raise ValueError(f'CSV contains no data rows: {path}')
    values = np.asarray(rows, dtype=float)
    gravity_m_s2 = None
    if gravity_values:
        gravity_array = np.asarray(gravity_values, dtype=float)
        valid_gravity = gravity_array[
            np.isfinite(gravity_array) & (gravity_array > 0)]
        if len(valid_gravity):
            gravity_m_s2 = float(np.median(valid_gravity))
    return FlightData(
        path=path,
        acc_x=values[:, 0], acc_y=values[:, 1],
        vis_x=values[:, 2], vis_y=values[:, 3], height=values[:, 4],
        source_columns=source_columns, gravity_m_s2=gravity_m_s2)


def select_flight_window(flight, start, count):
    """Return the legacy five-column matrix for a validated sample window."""
    if start < 0:
        raise ValueError('start tick must be nonnegative')
    if count < 6:
        raise ValueError('tick count must be at least 6')
    if start + count > flight.row_count:
        raise ValueError(
            f'window {start}:{start + count} exceeds {flight.row_count} CSV rows')
    data = np.column_stack([
        flight.acc_x[start:start + count],
        flight.acc_y[start:start + count],
        flight.vis_x[start:start + count],
        flight.vis_y[start:start + count],
        flight.height[start:start + count],
    ])
    invalid = np.argwhere(~np.isfinite(data))
    if len(invalid):
        row, column = invalid[0]
        names = ('acc_x', 'acc_y', 'vis_x', 'vis_y', 'mheight')
        raise ValueError(
            f'non-numeric {names[column]} at CSV data row {start + row}')
    invalid_height = np.flatnonzero(data[:, 4] <= 0)
    if len(invalid_height):
        raise ValueError(
            f'mheight must be positive at CSV data row {start + invalid_height[0]}')
    return data


def resolve_gravity(flight, requested_gravity=None):
    """Use an explicit value, then the CSV G column, then standard gravity."""
    gravity = (requested_gravity if requested_gravity is not None
               else flight.gravity_m_s2)
    if gravity is None:
        gravity = STANDARD_GRAVITY
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError('gravity must be finite and positive')
    return float(gravity)


def integrate(acc):
    """Exact double integration of linearly interpolated acceleration, from rest."""
    acc = np.asarray(acc, dtype=float)
    velocity = np.zeros(len(acc), dtype=float)
    position = np.zeros(len(acc), dtype=float)
    if len(acc) > 1:
        velocity[1:] = np.cumsum(DT * (acc[:-1] + acc[1:]) / 2)
        position[1:] = np.cumsum(
            DT * velocity[:-1] + DT**2 * (acc[:-1] / 3 + acc[1:] / 6))
    return position


def trajectory_distance_metrics(estimate, reference):
    """Return RMSE and mean absolute distance between two trajectories."""
    difference = np.asarray(estimate) - np.asarray(reference)
    return (
        float(np.sqrt(np.mean(difference**2))),
        float(np.mean(np.abs(difference))),
    )


def average_2d_distance(estimate_x, reference_x, estimate_y, reference_y):
    """Return mean Euclidean distance between two X/Y trajectories."""
    dx = np.asarray(estimate_x) - np.asarray(reference_x)
    dy = np.asarray(estimate_y) - np.asarray(reference_y)
    return float(np.mean(np.hypot(dx, dy)))


def least_squares_affine_trend(t, values):
    """Return the fitted constant-offset plus constant-speed display trend."""
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if t.ndim != 1 or values.ndim != 1 or t.shape != values.shape:
        raise ValueError('trend time and values must be equal-length 1-D arrays')
    if len(t) == 0:
        raise ValueError('cannot fit a display trend to an empty trajectory')
    centered_t = t - np.mean(t)
    design = np.column_stack([np.ones(len(t)), centered_t])
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return design @ coefficients


def solve_normalized(design, target):
    """Least-squares solve with column normalization and fit diagnostics."""
    norms = np.linalg.norm(design, axis=0)
    norms[norms == 0] = 1
    coef, _, rank, singular = np.linalg.lstsq(design / norms, target, rcond=None)
    coef /= norms
    condition = float(singular[0] / singular[-1]) if singular[-1] else None
    return coef, int(rank), condition


def refit_motion(acc_without_bias, coordinate):
    """Re-estimate bias against an updated coordinate, then reintegrate."""
    t = np.arange(len(acc_without_bias)) * DT
    motion_design = np.column_stack([np.ones(len(t)), t, t**2 / 2])
    base_position = integrate(acc_without_bias)
    coef, rank, condition = solve_normalized(
        motion_design, coordinate - base_position)
    position0, velocity0, bias = coef
    corrected_acc = acc_without_bias + bias
    integrated_position = position0 + velocity0 * t + integrate(corrected_acc)
    residual = integrated_position - coordinate
    return ({'position0_m': float(position0),
             'velocity0_m_s': float(velocity0),
             'bias_m_s2': float(bias)},
            integrated_position, corrected_acc, residual, rank, condition)


def rebuild_state(acc, vis, height, gravity, scale, tilt_correction):
    """Update coordinate and acceleration, refit bias, and reintegrate."""
    delta = tilt_correction * height
    coordinate = scale * vis + delta
    # Keep the literal form from the experiment. Rebuilding from the raw
    # arrays prevents corrections from being applied twice across iterations.
    tilt_acceleration = delta * gravity / height
    physical_acc = acc + tilt_acceleration
    params, integrated_position, corrected_acc, residual, _, _ = refit_motion(
        physical_acc, coordinate)
    params['vision_scale'] = float(scale)
    params['tilt_correction_rad'] = float(tilt_correction)
    params['tilt_correction_deg'] = float(np.degrees(tilt_correction))
    params['net_acceleration_correction_m_s2'] = float(
        params['bias_m_s2'] + gravity * tilt_correction)
    params['fit_rmse_m'] = float(np.sqrt(np.mean(residual**2)))
    return (params, integrated_position, coordinate, physical_acc, corrected_acc,
            delta, tilt_acceleration, residual)


def history_entry(iteration, update, params):
    return {'iteration': iteration, 'update': update,
            'vision_scale': params['vision_scale'],
            'tilt_correction_rad': params['tilt_correction_rad'],
            'bias_m_s2': params['bias_m_s2'],
            'net_acceleration_correction_m_s2':
                params['net_acceleration_correction_m_s2'],
            'fit_rmse_m': params['fit_rmse_m']}


def iterative_true_fit(acc, vis, height, gravity, scale, gain, iterations,
                       acceleration_sign):
    """Apply residual corrections to coordinate and acceleration repeatedly."""
    scaled_vision = scale * vis.copy()
    initial_coordinate = scaled_vision.copy()
    coordinate = scaled_vision.copy()
    physical_acc = acc.copy()
    (params, integrated_position, corrected_acc, residual,
     _, _) = refit_motion(physical_acc, coordinate)
    roughness = float(np.sqrt(np.mean(np.diff(coordinate, n=2)**2)))
    fit_rmse = float(np.sqrt(np.mean(residual**2)))
    scaled_rmse, scaled_mae = trajectory_distance_metrics(
        integrated_position, scaled_vision)
    original_rmse, original_mae = trajectory_distance_metrics(
        integrated_position, vis)
    history = [{'iteration': 0,
                'bias_m_s2': params['bias_m_s2'],
                'rmse_m': scaled_rmse,
                'scaled_vision_rmse_m': scaled_rmse,
                'scaled_vision_mean_abs_distance_m': scaled_mae,
                'fit_rmse_m': fit_rmse,
                'original_vision_rmse_m': original_rmse,
                'original_vision_mean_abs_distance_m': original_mae,
                'coordinate_update_rms_m': 0.0,
                'max_abs_coordinate_update_over_height': 0.0,
                'max_abs_total_coordinate_change_over_height': 0.0,
                'rmse_improved_vs_previous': None,
                'scaled_vision_rmse_improved_vs_previous': None,
                'fit_rmse_improved_vs_previous': None,
                'original_vision_rmse_improved_vs_previous': None,
                'coordinate_second_difference_rms_m': roughness}]
    states = [{'iteration': 0, 'coordinate': coordinate.copy(),
               'coordinate_delta': np.zeros_like(coordinate),
               'physical_acc': physical_acc.copy(),
               'acceleration_delta': np.zeros_like(physical_acc),
               'corrected_acc': corrected_acc.copy(),
               'integrated_position': integrated_position.copy(),
               'bias_m_s2': params['bias_m_s2'],
               'max_abs_total_coordinate_change_over_height': 0.0,
               'rmse_m': scaled_rmse,
               'scaled_vision_rmse_m': scaled_rmse,
               'scaled_vision_mean_abs_distance_m': scaled_mae,
               'fit_rmse_m': fit_rmse,
               'original_vision_rmse_m': original_rmse,
               'original_vision_mean_abs_distance_m': original_mae,
               'roughness_m': roughness}]
    best_rmse = scaled_rmse
    best_iteration = 0
    best_bias = params['bias_m_s2']
    best_fit_rmse = fit_rmse
    best_fit_iteration = 0
    best_fit_bias = params['bias_m_s2']
    best_original_rmse = original_rmse
    best_original_iteration = 0
    best_original_bias = params['bias_m_s2']

    for iteration in range(1, iterations + 1):
        # Move vision toward the best current acceleration integration. Apply
        # the corresponding delta*G/height acceleration change, then refit the
        # bias and reintegrate before the next iteration.
        coordinate_delta = gain * (integrated_position - coordinate)
        coordinate = coordinate + coordinate_delta
        acceleration_delta = (acceleration_sign * coordinate_delta *
                              gravity / height)
        physical_acc = physical_acc + acceleration_delta
        (params, integrated_position, corrected_acc, residual,
         _, _) = refit_motion(physical_acc, coordinate)
        update_ratio = float(np.max(np.abs(coordinate_delta / height)))
        total_ratio = float(np.max(np.abs(
            (coordinate - initial_coordinate) / height)))
        fit_rmse = float(np.sqrt(np.mean(residual**2)))
        scaled_rmse, scaled_mae = trajectory_distance_metrics(
            integrated_position, scaled_vision)
        original_rmse, original_mae = trajectory_distance_metrics(
            integrated_position, vis)
        history.append({
            'iteration': iteration,
            'bias_m_s2': params['bias_m_s2'],
            'rmse_m': scaled_rmse,
            'scaled_vision_rmse_m': scaled_rmse,
            'scaled_vision_mean_abs_distance_m': scaled_mae,
            'fit_rmse_m': fit_rmse,
            'original_vision_rmse_m': original_rmse,
            'original_vision_mean_abs_distance_m': original_mae,
            'coordinate_update_rms_m': float(
                np.sqrt(np.mean(coordinate_delta**2))),
            'max_abs_coordinate_update_over_height': update_ratio,
            'max_abs_total_coordinate_change_over_height': total_ratio,
            'rmse_improved_vs_previous':
                scaled_rmse < history[-1]['rmse_m'],
            'scaled_vision_rmse_improved_vs_previous':
                scaled_rmse < history[-1]['scaled_vision_rmse_m'],
            'fit_rmse_improved_vs_previous':
                fit_rmse < history[-1]['fit_rmse_m'],
            'original_vision_rmse_improved_vs_previous':
                original_rmse < history[-1]['original_vision_rmse_m'],
            'coordinate_second_difference_rms_m': float(
                np.sqrt(np.mean(np.diff(coordinate, n=2)**2)))})
        states.append({
            'iteration': iteration, 'coordinate': coordinate.copy(),
            'coordinate_delta': coordinate_delta.copy(),
            'physical_acc': physical_acc.copy(),
            'acceleration_delta': acceleration_delta.copy(),
            'corrected_acc': corrected_acc.copy(),
            'integrated_position': integrated_position.copy(),
            'bias_m_s2': params['bias_m_s2'],
            'max_abs_total_coordinate_change_over_height': total_ratio,
            'rmse_m': scaled_rmse,
            'scaled_vision_rmse_m': scaled_rmse,
            'scaled_vision_mean_abs_distance_m': scaled_mae,
            'fit_rmse_m': fit_rmse,
            'original_vision_rmse_m': original_rmse,
            'original_vision_mean_abs_distance_m': original_mae,
            'roughness_m': history[-1]['coordinate_second_difference_rms_m']})
        if scaled_rmse < best_rmse:
            best_rmse = scaled_rmse
            best_iteration = iteration
            best_bias = params['bias_m_s2']
        if fit_rmse < best_fit_rmse:
            best_fit_rmse = fit_rmse
            best_fit_iteration = iteration
            best_fit_bias = params['bias_m_s2']
        if original_rmse < best_original_rmse:
            best_original_rmse = original_rmse
            best_original_iteration = iteration
            best_original_bias = params['bias_m_s2']

    max_ratio = max(
        item['max_abs_total_coordinate_change_over_height'] for item in history)
    summary = {
        'vision_scale': float(scale),
        'iterations': iterations,
        'gain': gain,
        'acceleration_sign': acceleration_sign,
        'initial_rmse_m': history[0]['rmse_m'],
        'last_iteration_rmse_m': history[-1]['rmse_m'],
        'best_rmse_m': best_rmse,
        'best_iteration': best_iteration,
        'metric_note': ('rmse_m compares integrated position with the initial '
                        'scaled vision; fit_rmse_m is the internal fit to the '
                        'iteratively updated coordinate'),
        'initial_scaled_vision_rmse_m':
            history[0]['scaled_vision_rmse_m'],
        'last_iteration_scaled_vision_rmse_m':
            history[-1]['scaled_vision_rmse_m'],
        'last_iteration_scaled_vision_mean_abs_distance_m':
            history[-1]['scaled_vision_mean_abs_distance_m'],
        'best_scaled_vision_rmse_m': best_rmse,
        'best_scaled_vision_iteration': best_iteration,
        'best_scaled_vision_bias_m_s2': best_bias,
        'initial_fit_rmse_m': history[0]['fit_rmse_m'],
        'last_iteration_fit_rmse_m': history[-1]['fit_rmse_m'],
        'best_fit_rmse_m': best_fit_rmse,
        'best_fit_iteration': best_fit_iteration,
        'best_fit_bias_m_s2': best_fit_bias,
        'initial_original_vision_rmse_m':
            history[0]['original_vision_rmse_m'],
        'last_iteration_original_vision_rmse_m':
            history[-1]['original_vision_rmse_m'],
        'last_iteration_original_vision_mean_abs_distance_m':
            history[-1]['original_vision_mean_abs_distance_m'],
        'best_original_vision_rmse_m': best_original_rmse,
        'best_original_vision_iteration': best_original_iteration,
        'best_original_vision_bias_m_s2': best_original_bias,
        'best_bias_m_s2': best_bias,
        'selected_iteration': history[-1]['iteration'],
        'selection': 'requested final iteration',
        'final_bias_m_s2': params['bias_m_s2'],
        'total_coordinate_change_rms_m': float(
            np.sqrt(np.mean((coordinate - initial_coordinate)**2))),
        'max_abs_final_acc_m_s2': float(np.max(np.abs(corrected_acc))),
        'max_abs_total_coordinate_change_over_height': max_ratio,
        'max_small_angle_linearization_overestimate_percent':
            100.0 * (np.sqrt(1.0 + max_ratio**2) - 1.0),
        'small_angle_ratio_warning_threshold': SMALL_ANGLE_RATIO_WARNING,
        'small_angle_assumption_exceeded':
            max_ratio > SMALL_ANGLE_RATIO_WARNING,
        'rmse_improved_every_iteration': all(
            item['rmse_improved_vs_previous'] is not False for item in history),
        'scaled_vision_rmse_improved_every_iteration': all(
            item['scaled_vision_rmse_improved_vs_previous'] is not False
            for item in history),
        'fit_rmse_improved_every_iteration': all(
            item['fit_rmse_improved_vs_previous'] is not False
            for item in history),
        'original_vision_rmse_improved_every_iteration': all(
            item['original_vision_rmse_improved_vs_previous'] is not False
            for item in history),
        'history': history,
    }
    return (coordinate, corrected_acc, integrated_position, summary, states)


def compute_feedback_preview(flight, start, count, scale_percent, gravity=None,
                             gain=0.3, iterations=4, acceleration_sign=1):
    """Compute one scale/window for the interactive preview without writing files."""
    started = time.perf_counter()
    if not np.isfinite(scale_percent) or scale_percent <= 0:
        raise ValueError('vision scale must be finite and positive')
    gravity_source = ('explicit' if gravity is not None else
                      ('CSV G' if flight.gravity_m_s2 is not None
                       else 'standard fallback'))
    gravity = resolve_gravity(flight, gravity)
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError('correction gain must be in (0, 1]')
    if iterations < 1:
        raise ValueError('correction iterations must be positive')
    if acceleration_sign not in (-1, 1):
        raise ValueError('acceleration sign must be -1 or +1')

    data = select_flight_window(flight, int(start), int(count))
    axes = {}
    for axis_index, axis in enumerate(('x', 'y')):
        (_, _, _, summary, states) = iterative_true_fit(
            data[:, axis_index], data[:, axis_index + 2], data[:, 4], gravity,
            scale_percent / 100.0, gain, iterations, acceleration_sign)
        axes[axis] = {
            'raw_acc': data[:, axis_index].copy(),
            'original_vision': data[:, axis_index + 2].copy(),
            'scaled_vision': (
                scale_percent / 100.0 * data[:, axis_index + 2]).copy(),
            'states': states,
            'summary': summary,
        }
    return {
        't': np.arange(int(count)) * DT,
        'start': int(start),
        'count': int(count),
        'scale_percent': float(scale_percent),
        'average_height_m': float(np.mean(data[:, 4])),
        'gravity_m_s2': gravity,
        'gravity_source': gravity_source,
        'compute_seconds': time.perf_counter() - started,
        'axes': axes,
    }


def fit_axis(acc, vis, height, gravity, stage, tilt, previous_scale=1.0,
             max_iterations=50, tolerance=1e-9, fit_tilt=False,
             max_tilt_rad=None):
    """Run a stage, rebuilding all corrected arrays after every parameter update."""
    t = np.arange(len(acc)) * DT
    base = integrate(acc)
    constant_position = t**2 / 2
    gravity_position = gravity * constant_position
    motion_columns = [np.ones(len(t)), t, constant_position]
    scale = previous_scale
    tilt_correction = 0.0
    history = []

    if stage == 1:
        fit_design = np.column_stack(motion_columns)
        result = rebuild_state(acc, vis, height, gravity, scale, tilt_correction)
        fit_rank, fit_condition = solve_normalized(fit_design, vis - base)[1:]
        history.append(history_entry(0, 'bias', result[0]))
    elif stage == 2:
        # Scale update, followed immediately by coordinate rebuilding, bias
        # refitting, acceleration rebuilding, and reintegration.
        fit_design = np.column_stack(motion_columns + [-vis])
        coef, fit_rank, fit_condition = solve_normalized(fit_design, -base)
        scale = float(coef[3])
        result = rebuild_state(acc, vis, height, gravity, scale, tilt_correction)
        history.append(history_entry(0, 'scale', result[0]))
    else:
        # Alternating least squares. Each half-step also solves provisional
        # motion terms, then rebuild_state explicitly re-estimates bias using
        # the updated coordinate and physical acceleration before integrating.
        free_tilt = tilt is None and fit_tilt
        fixed_tilt = 0.0 if tilt is None else tilt
        previous = rebuild_state(
            acc, vis, height, gravity, scale, fixed_tilt)
        converged = False
        completed_iterations = 0
        tilt_bound_hit = False
        for iteration in range(1, max_iterations + 1):
            old_scale = scale
            old_tilt = previous[0]['tilt_correction_rad']
            old_bias = previous[0]['bias_m_s2']

            tilt_correction = old_tilt if free_tilt else fixed_tilt
            scale_design = np.column_stack(motion_columns + [-vis])
            scale_target = -base - tilt_correction * (gravity_position - height)
            scale_coef, _, _ = solve_normalized(scale_design, scale_target)
            scale = float(scale_coef[3])
            result = rebuild_state(
                acc, vis, height, gravity, scale, tilt_correction)
            history.append(history_entry(iteration, 'scale', result[0]))

            if free_tilt:
                tilt_design = np.column_stack(
                    motion_columns + [gravity_position - height])
                tilt_coef, _, _ = solve_normalized(
                    tilt_design, scale * vis - base)
                tilt_correction = float(tilt_coef[3])
                if max_tilt_rad is not None:
                    limited_tilt = float(np.clip(
                        tilt_correction, -max_tilt_rad, max_tilt_rad))
                    tilt_bound_hit |= limited_tilt != tilt_correction
                    tilt_correction = limited_tilt
                result = rebuild_state(
                    acc, vis, height, gravity, scale, tilt_correction)
                history.append(history_entry(iteration, 'tilt', result[0]))

            completed_iterations = iteration
            parameter_change = max(
                abs(scale - old_scale), abs(tilt_correction - old_tilt),
                abs(result[0]['bias_m_s2'] - old_bias))
            previous = result
            if parameter_change <= tolerance:
                converged = True
                break
            if not free_tilt:
                # With zero or supplied tilt, the scale/bias update is an exact
                # linear solve, so another identical iteration is unnecessary.
                converged = True
                break

        fit_design = np.column_stack(
            motion_columns + [-vis] +
            ([gravity_position - height] if free_tilt else []))
        fit_target = (-base - tilt_correction * (gravity_position - height)
                      if not free_tilt else -base)
        _, fit_rank, fit_condition = solve_normalized(fit_design, fit_target)
        result[0]['iterations'] = completed_iterations
        result[0]['converged'] = converged
        if free_tilt:
            result[0]['tilt_mode'] = ('bounded_fit' if max_tilt_rad is not None
                                      else 'unconstrained_fit')
            result[0]['tilt_bound_hit'] = tilt_bound_hit
        elif tilt is not None:
            result[0]['tilt_mode'] = 'supplied'
            result[0]['supplied_tilt_correction_rad'] = tilt
        else:
            result[0]['tilt_mode'] = 'disabled'
        result[0]['vision_correction_span_m'] = float(np.ptp(result[5]))
        result[0]['raw_vision_span_m'] = float(np.ptp(vis))

    params = result[0]
    params.update(rank=int(fit_rank), parameters=fit_design.shape[1],
                  condition=fit_condition)
    return (*result, history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', nargs='?', type=Path,
                        help='CSV file; omit to use height.csv in interactive mode')
    parser.add_argument('start', nargs='?', type=int,
                        help='Zero-based data row; omit with count to open the UI')
    parser.add_argument('count', nargs='?', type=int,
                        help='Sample count; omit with start to open the UI')
    parser.add_argument(
        '--gravity', type=float,
        help='G in m/s^2 (default: CSV G column, otherwise 9.80665)')
    parser.add_argument('--tilt-x', type=float, help='Known signed X tilt correction in radians')
    parser.add_argument('--tilt-y', type=float, help='Known signed Y tilt correction in radians')
    parser.add_argument('--fit-tilt', action='store_true',
                        help='Experimentally fit missing tilt values (disabled by default)')
    parser.add_argument('--max-tilt-deg', type=float,
                        help='Symmetric tilt bound in degrees; requires --fit-tilt')
    parser.add_argument('--max-iterations', type=int, default=50,
                        help='Maximum scale/tilt refit cycles (default: 50)')
    parser.add_argument('--tolerance', type=float, default=1e-9,
                        help='Parameter convergence tolerance (default: 1e-9)')
    parser.add_argument('--correction-gain', type=float, default=0.3,
                        help='Fraction of integration/vision difference applied per iteration')
    parser.add_argument('--correction-iterations', type=int, default=4,
                        help='Residual correction iterations for true_PERCENT columns '
                             '(default: 4)')
    parser.add_argument('--acceleration-sign', type=int, choices=(-1, 1), default=1,
                        help='+1 matches the NED geometry; -1 is diagnostic only')
    parser.add_argument('--scale-start', type=int, default=70)
    parser.add_argument('--scale-stop', type=int, default=120)
    parser.add_argument('--scale-step', type=int, default=5)
    parser.add_argument('--plot-scale', type=int, default=100,
                        help='Percentage whose correction iterations are shown in the main plot')
    parser.add_argument('--output', type=Path, help='Output filename prefix')
    parser.add_argument('--write-csv', action='store_true',
                        help='Write batch CSV exports (disabled by default)')
    parser.add_argument('--no-show', action='store_true', help='Save plots without opening a window')
    args = parser.parse_args()
    if (args.gravity is not None and
            (not np.isfinite(args.gravity) or args.gravity <= 0)):
        parser.error('gravity must be finite and positive')
    if args.max_iterations < 1:
        parser.error('max-iterations must be positive')
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('tolerance must be finite and positive')
    if not np.isfinite(args.correction_gain) or not 0 < args.correction_gain <= 1:
        parser.error('correction-gain must be in (0, 1]')
    if args.correction_iterations < 1:
        parser.error('correction-iterations must be positive')

    if (args.start is None) != (args.count is None):
        parser.error('start and count must either both be supplied or both be omitted')
    if args.start is None:
        if args.no_show:
            parser.error('--no-show requires the batch form: CSV START COUNT')
        from imuviso_ui import launch_interactive
        launch_interactive(
            initial_path=resolve_default_csv(args.csv_path),
            initial_start=0,
            initial_count=500,
            initial_scale=args.plot_scale,
            gravity=args.gravity,
            gain=args.correction_gain,
            iterations=args.correction_iterations,
            acceleration_sign=args.acceleration_sign)
        return

    if args.csv_path is None:
        parser.error('batch mode requires CSV START COUNT')
    if args.start < 0 or args.count < 6:
        parser.error('start must be nonnegative and count must be at least 6')
    if args.scale_step <= 0 or args.scale_start > args.scale_stop:
        parser.error('scale range must be ascending with a positive step')
    if args.scale_start <= 0 or args.scale_stop <= 0:
        parser.error('scale percentages must be positive')
    scale_percentages = list(range(
        args.scale_start, args.scale_stop + 1, args.scale_step))
    if args.plot_scale not in scale_percentages:
        parser.error('plot-scale must be one of the generated scale percentages')
    if args.max_tilt_deg is not None:
        if not args.fit_tilt:
            parser.error('max-tilt-deg requires --fit-tilt')
        if not np.isfinite(args.max_tilt_deg) or args.max_tilt_deg <= 0:
            parser.error('max-tilt-deg must be finite and positive')
    for tilt in (args.tilt_x, args.tilt_y):
        if tilt is not None and not np.isfinite(tilt):
            parser.error('tilt must be finite')
    if args.acceleration_sign != 1:
        print('WARNING: acceleration-sign=-1 contradicts the configured NED '
              'geometry; positive DX should produce positive DX*G/height.')
    try:
        flight = load_flight_csv(args.csv_path)
        data = select_flight_window(flight, args.start, args.count)
        gravity = resolve_gravity(flight, args.gravity)
        acc_names = [flight.source_columns['acc_x'], flight.source_columns['acc_y']]
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    prefix = args.output or flight.path.with_name(
        f'{flight.path.stem}_fit_{args.start}_{args.count}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(args.count) * DT
    report = {'start': args.start, 'count': args.count, 'dt_s': DT, 'gravity_m_s2': gravity,
              'gravity_source': ('--gravity' if args.gravity is not None
                                 else ('CSV G' if flight.gravity_m_s2 is not None
                                       else 'standard fallback')),
              'model': 'integrated(acc + bias + G*tilt_correction) = scale*vis + tilt_correction*mheight',
              'update_order': ['update vision coordinate', 'update physical acceleration',
                               're-estimate acceleration bias', 'reintegrate acceleration'],
              'max_iterations': args.max_iterations,
              'tolerance': args.tolerance,
              'fit_tilt': args.fit_tilt,
              'max_tilt_deg': args.max_tilt_deg,
              'residual_correction': {
                  'gain': args.correction_gain,
                  'iterations': args.correction_iterations,
                  'acceleration_sign': args.acceleration_sign,
                  'geometry': {
                      'coordinate_delta':
                          'DX = gain * (integrated_position - coordinate)',
                      'coordinate_update': 'coordinate = coordinate + DX',
                      'acceleration_update':
                          'physical_acceleration += acceleration_sign * DX*G/height',
                      'relationship':
                          ('positive DX gives positive acceleration correction'
                           if args.acceleration_sign == 1 else
                           'diagnostic inversion: positive DX gives negative correction'),
                      'approximation': 'small angle: abs(DX/height) << 1',
                      'matches_diagram': args.acceleration_sign == 1,
                  },
                  'small_angle_ratio_warning_threshold':
                      SMALL_ANGLE_RATIO_WARNING,
                  'scale_percentages': scale_percentages,
                  'plot_scale': args.plot_scale,
                  'source_columns': {'x_acceleration': acc_names[0],
                                     'y_acceleration': acc_names[1],
                                     'x_coordinate': 'vis_x',
                                     'y_coordinate': 'vis_y',
                                     'height': 'mheight'},
                  'fits': {}},
              'note': ('Unknown tilt is disabled by default because free bias '
                       'absorbs its acceleration effect. Supply --tilt-x/--tilt-y, '
                       'or explicitly use --fit-tilt. Every update rebuilds '
                       'coordinate and acceleration, refits bias, and '
                       'reintegrates. Unqualified RMSE fields compare the '
                       'integrated trajectory with the initial scaled vision; '
                       'fit_rmse fields compare with the iteratively adjusted '
                       'coordinate.'),
              'fits': {}, 'iteration_history': {}}
    output_columns = {'sample': np.arange(args.start, args.start + args.count),
                      'time_s': t, 'mheight': data[:, 4]}
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey='col')
    stage_names = {1: 'acceleration bias', 2: '+ vision scale', 3: '+ tilt/height'}
    for axis_index, axis in enumerate(('x', 'y')):
        acc, vis = data[:, axis_index], data[:, axis_index + 2]
        output_columns[f'acc_{axis}'] = acc
        output_columns[f'vis_{axis}'] = vis
        stage2_scale = 1.0
        for stage in (1, 2, 3):
            tilt = (args.tilt_x if axis == 'x' else args.tilt_y) if stage == 3 else None
            (params, predicted, calibrated_vis, physical_acc, corrected_acc,
             delta, tilt_acceleration, residual, history) = fit_axis(
                acc, vis, data[:, 4], gravity, stage, tilt, stage2_scale,
                args.max_iterations, args.tolerance, args.fit_tilt,
                (np.radians(args.max_tilt_deg)
                 if args.max_tilt_deg is not None else None))
            scaled_vis = params['vision_scale'] * vis
            scaled_rmse, scaled_mae = trajectory_distance_metrics(
                predicted, scaled_vis)
            original_rmse, original_mae = trajectory_distance_metrics(
                predicted, vis)
            params['rmse_m'] = scaled_rmse
            params['scaled_vision_rmse_m'] = scaled_rmse
            params['scaled_vision_mean_abs_distance_m'] = scaled_mae
            params['original_vision_rmse_m'] = original_rmse
            params['original_vision_mean_abs_distance_m'] = original_mae
            params['rmse_metric_note'] = (
                'fit_rmse_m uses the adjusted coordinate; '
                'rmse_m and scaled_vision_rmse_m use initial scaled vision; '
                'original_vision_rmse_m uses raw vision')
            if stage == 2:
                stage2_scale = params['vision_scale']
            key = f'stage{stage}_{axis}'
            report['fits'][key] = params
            report['iteration_history'][key] = history
            for name, values in [('corrected_position', predicted),
                                 ('calibrated_vision', calibrated_vis),
                                 ('physical_acceleration', physical_acc),
                                 ('corrected_acceleration', corrected_acc),
                                 ('vision_delta', delta),
                                 ('tilt_acceleration', tilt_acceleration),
                                 ('fit_residual', residual),
                                 ('scaled_vision_residual',
                                  predicted - scaled_vis),
                                 ('original_vision_residual', predicted - vis)]:
                output_columns[f'{key}_{name}'] = values
            ax = axes[stage - 1, axis_index]
            raw_label = ('Raw vision / scale 100% reference'
                         if stage == 1 else 'Raw vision')
            ax.plot(t, vis, color='0.7', linewidth=1, label=raw_label)
            if stage >= 2:
                ax.plot(t, scaled_vis, color='tab:green', linewidth=1.2,
                        linestyle='--',
                        label='Scaled vision (RMSE reference)')
                if np.any(delta != 0):
                    ax.plot(t, calibrated_vis, color='tab:blue', linewidth=1,
                            label='Scale/tilt-adjusted vision')
            ax.plot(t, predicted, color='tab:orange', linewidth=2,
                    label='Smooth corrected trajectory')
            iteration_text = ''
            if stage == 3:
                if params['tilt_mode'] == 'disabled':
                    iteration_text = ', tilt disabled'
                elif params['tilt_mode'] == 'supplied':
                    iteration_text = ', supplied tilt'
                else:
                    iteration_text = (f', {params["tilt_mode"]}, '
                                      f'{params["iterations"]} cycles')
            ax.set_title(
                f'{axis.upper()} stage {stage}: {stage_names[stage]}\n'
                f'scaled-vision RMSE {scaled_rmse:.4g} m; '
                f'mean |error| {scaled_mae:.4g} m\n'
                f'internal adjusted-coordinate RMSE '
                f'{params["fit_rmse_m"]:.4g} m{iteration_text}',
                fontsize='medium')
            ax.set_ylabel('Position (m)')
            ax.grid(True)
            ax.legend(loc='upper left', fontsize='small')
            print(f'{key}: {json.dumps(params)}')
            if params['rank'] < params['parameters']:
                print(f'WARNING: {key} is rank deficient; parameters are not uniquely determined.')
            elif params['condition'] > 1e6:
                print(f'WARNING: {key} is poorly conditioned; parameter estimates may be unstable.')
            if params.get('vision_scale', 1) <= 0:
                print(f'WARNING: {key} has a nonpositive vision scale; inspect this window and model.')
            if stage == 3 and not params['converged']:
                print(f'WARNING: {key} did not converge in {params["iterations"]} iterations.')
            if (stage == 3 and params['tilt_mode'].endswith('fit') and
                    params['vision_correction_span_m'] > params['raw_vision_span_m']):
                print(f'WARNING: {key} tilt correction spans '
                      f'{params["vision_correction_span_m"]:.3g} m, larger than the raw '
                      f'vision span {params["raw_vision_span_m"]:.3g} m.')
    for ax in axes[-1]:
        ax.set_xlabel('Time from window start (s)')
    fig.suptitle('Calibration stages: measurements and IMU-fitted trajectory')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    calibration_png_path = Path(str(prefix) + '_calibration.png')
    fig.savefig(calibration_png_path, dpi=160)

    # Prompt steps 4-6: make independent residual-correction experiments for
    # every requested initial vision scale and save their final coordinates
    # and accelerations alongside the original NED inputs.
    true_columns = {
        'sample': np.arange(args.start, args.start + args.count),
        'time_s': t,
        'original_ned_x': data[:, 2],
        'original_ned_y': data[:, 3],
        'original_acc_x': data[:, 0],
        'original_acc_y': data[:, 1],
        'mheight': data[:, 4],
    }
    plot_iteration_states = {}
    for percent in scale_percentages:
        scale = percent / 100.0
        percent_results = {}
        for axis_index, axis in enumerate(('x', 'y')):
            (coordinate, corrected_acc, integrated_position,
             summary, states) = iterative_true_fit(
                data[:, axis_index], data[:, axis_index + 2], data[:, 4],
                gravity, scale, args.correction_gain,
                args.correction_iterations, args.acceleration_sign)
            percent_results[axis] = (
                coordinate, corrected_acc, integrated_position,
                states[-1]['physical_acc'])
            report['residual_correction']['fits'][f'{percent}_{axis}'] = summary
            if percent == args.plot_scale:
                plot_iteration_states[axis] = states
                if summary['small_angle_assumption_exceeded']:
                    print(
                        f'WARNING: {percent}% {axis.upper()} feedback reaches '
                        f'|DX/height|={summary["max_abs_total_coordinate_change_over_height"]:.3g}; '
                        f'the DX*G/height small-angle approximation is intended '
                        f'for values well below 1 (worst-case linearized '
                        f'correction is about '
                        f'{summary["max_small_angle_linearization_overestimate_percent"]:.2g}% high).')
                if not summary[
                        'scaled_vision_rmse_improved_every_iteration']:
                    print(
                        f'WARNING: {percent}% {axis.upper()} feedback '
                        f'initial-scaled-vision RMSE did '
                        f'not improve on every iteration; best iteration is '
                        f'{summary["best_scaled_vision_iteration"]}, '
                        f'final iteration is '
                        f'{summary["selected_iteration"]}.')
        true_columns[f'true_{percent}_x'] = percent_results['x'][0]
        true_columns[f'true_{percent}_y'] = percent_results['y'][0]
        true_columns[f'true_{percent}_scaled_vision_x'] = scale * data[:, 2]
        true_columns[f'true_{percent}_scaled_vision_y'] = scale * data[:, 3]
        true_columns[f'true_{percent}_physical_acc_x'] = percent_results['x'][3]
        true_columns[f'true_{percent}_physical_acc_y'] = percent_results['y'][3]
        true_columns[f'true_{percent}_acc_x'] = percent_results['x'][1]
        true_columns[f'true_{percent}_acc_y'] = percent_results['y'][1]
        true_columns[f'true_{percent}_integrated_x'] = percent_results['x'][2]
        true_columns[f'true_{percent}_integrated_y'] = percent_results['y'][2]

    true_csv_path = Path(str(prefix) + '_true.csv')
    if args.write_csv:
        with true_csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(true_columns)
            writer.writerows(zip(*true_columns.values()))

    iteration_rows = 2
    iteration_fig, iteration_axes = plt.subplots(
        iteration_rows, 2, figsize=(14, 3.6 * iteration_rows),
        sharex=True, sharey='col', squeeze=False)
    for axis_index, axis in enumerate(('x', 'y')):
        states = plot_iteration_states[axis]
        displayed_states = (states[0], states[-1])
        original_vision = data[:, axis_index + 2]
        scaled_vision = args.plot_scale / 100.0 * original_vision
        initial_roughness = states[0]['roughness_m']
        raw_display_trend = least_squares_affine_trend(t, original_vision)
        coordinate_display_trend = least_squares_affine_trend(
            t, states[0]['coordinate'])
        for row_index, state in enumerate(displayed_states):
            ax = iteration_axes[row_index, axis_index]
            comparison_center = (
                scaled_vision + state['integrated_position']) / 2.0
            # Plot-only detrending: one shared line keeps pairwise errors exact.
            comparison_display_trend = least_squares_affine_trend(
                t, comparison_center)
            ax.plot(t, original_vision - raw_display_trend,
                    color='0.25', linewidth=0.9, linestyle=':',
                    marker='.', markersize=2.5, markevery=10, zorder=4,
                    label='Raw vision (dotted; overlaps scaled at 100%)')
            ax.plot(t, scaled_vision - comparison_display_trend,
                    color='tab:green', linewidth=1.2,
                    linestyle='--', zorder=3,
                    label='Initial scaled vision (RMSE reference)')
            if row_index > 0:
                ax.plot(t, state['coordinate'] - coordinate_display_trend,
                        color='tab:blue', linewidth=1,
                        label=(f'Blended vision coordinate, iteration '
                               f'{state["iteration"]}'))
                ax.plot(t, (states[0]['integrated_position']
                            - comparison_display_trend),
                        color='tab:purple', linewidth=1.4, linestyle='--',
                        label='Initial rebiased acceleration integration')
            integration_label = (
                'Initial rebiased acceleration integration'
                if row_index == 0 else
                'Final rebiased acceleration integration')
            ax.plot(t, (state['integrated_position']
                        - comparison_display_trend), color='tab:orange',
                    linewidth=2, label=integration_label)
            roughness_percent = (
                100 * state['roughness_m'] / initial_roughness
                if initial_roughness > 0 else 0.0)
            ax.set_title(
                f'{axis.upper()} iter {state["iteration"]}: '
                f'integration vs initial scaled vision\n'
                f'absolute RMSE {state["scaled_vision_rmse_m"]:.4g} m; '
                f'mean |error| '
                f'{state["scaled_vision_mean_abs_distance_m"]:.4g} m\n'
                f'bias {state["bias_m_s2"]:.4g} m/s²\n'
                f'vision roughness {state["roughness_m"]:.4g} m '
                f'({roughness_percent:.1f}% of iter 0)\n'
                f'max |total DX/H| '
                f'{state["max_abs_total_coordinate_change_over_height"]:.3g}')
            ax.set_ylabel('Detrended position (m)')
            ax.grid(True)
            ax.legend(loc='upper left', fontsize='small')
    for ax in iteration_axes[-1]:
        ax.set_xlabel('Time from window start (s)')
    plot_average_2d_distance = average_2d_distance(
        plot_iteration_states['x'][-1]['integrated_position'],
        args.plot_scale / 100.0 * data[:, 2],
        plot_iteration_states['y'][-1]['integrated_position'],
        args.plot_scale / 100.0 * data[:, 3])
    iteration_fig.suptitle(
        f'Vision/acceleration feedback at {args.plot_scale}% initial vision scale '
        f'(gain {args.correction_gain:g}); absolute final average 2D '
        f'integration-to-initial-scaled-vision distance '
        f'{plot_average_2d_distance:.4g} m\n'
        f'Display only: offset and constant speed removed separately; '
        f'exported values remain absolute')
    iteration_fig.tight_layout(rect=(0, 0, 1, 0.92))
    main_png_path = Path(str(prefix) + '.png')
    iteration_fig.savefig(main_png_path, dpi=160)

    acceleration_fig, acceleration_axes = plt.subplots(
        iteration_rows, 2, figsize=(14, 3.2 * iteration_rows),
        sharex=True, sharey='col', squeeze=False)
    for axis_index, axis in enumerate(('x', 'y')):
        states = plot_iteration_states[axis]
        displayed_states = (states[0], states[-1])
        raw_acc = data[:, axis_index]
        for row_index, state in enumerate(displayed_states):
            ax = acceleration_axes[row_index, axis_index]
            if row_index > 0:
                ax.plot(t, raw_acc, color='0.75', linewidth=1,
                        label='Original raw acceleration')
                physical_label = 'Acceleration after residual update'
            else:
                physical_label = 'Original raw acceleration'
            ax.plot(t, state['physical_acc'], color='tab:blue', linewidth=1,
                    label=physical_label)
            ax.plot(t, state['corrected_acc'], color='tab:orange', linewidth=1.5,
                    label='Acceleration after bias refit')
            update_rms = float(np.sqrt(np.mean(
                (state['physical_acc'] - raw_acc)**2)))
            ax.set_title(
                f'{axis.upper()} iteration {state["iteration"]}: '
                f'acceleration update RMS {update_rms:.4g} m/s²\n'
                f'refitted bias {state["bias_m_s2"]:.4g} m/s²')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.grid(True)
            ax.legend(loc='upper left')
    for ax in acceleration_axes[-1]:
        ax.set_xlabel('Time from window start (s)')
    acceleration_fig.suptitle(
        f'Acceleration feedback at {args.plot_scale}% initial vision scale '
        f'(gain {args.correction_gain:g})')
    acceleration_fig.tight_layout()
    acceleration_png_path = Path(str(prefix) + '_acceleration.png')
    acceleration_fig.savefig(acceleration_png_path, dpi=160)

    Path(str(prefix) + '.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.write_csv:
        with Path(str(prefix) + '.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(output_columns)
            writer.writerows(zip(*output_columns.values()))
    print(f'Saved {prefix}.json, {main_png_path}, '
          f'{calibration_png_path} and {acceleration_png_path}')
    if args.write_csv:
        print(f'Saved {prefix}.csv and {true_csv_path}')
    if not args.no_show:
        plt.show()
    plt.close(fig)
    plt.close(iteration_fig)
    plt.close(acceleration_fig)


if __name__ == '__main__':
    main()
