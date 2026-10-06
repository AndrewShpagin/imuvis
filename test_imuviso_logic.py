import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import imuviso


class FeedbackGeometryTests(unittest.TestCase):
    def test_display_detrending_preserves_comparison_differences(self):
        t = np.linspace(0.0, 12.0, 241)
        scaled = 4.0 + 0.25 * t + 0.3 * np.sin(t)
        integrated = -1.0 + 0.55 * t + 0.2 * np.cos(0.7 * t)
        scaled_before = scaled.copy()
        integrated_before = integrated.copy()

        midpoint = (scaled + integrated) / 2.0
        trend = imuviso.least_squares_affine_trend(t, midpoint)
        displayed_scaled = scaled - trend
        displayed_integrated = integrated - trend

        np.testing.assert_allclose(
            displayed_integrated - displayed_scaled,
            integrated - scaled,
        )
        centered_t = t - np.mean(t)
        displayed_midpoint = (
            displayed_scaled + displayed_integrated) / 2.0
        self.assertAlmostEqual(float(np.mean(displayed_midpoint)), 0.0)
        self.assertAlmostEqual(
            float(np.dot(centered_t, displayed_midpoint)), 0.0,
            places=10,
        )

        common_line = 8.0 - 0.4 * t
        shifted_midpoint = (
            scaled + common_line + integrated + common_line) / 2.0
        shifted_trend = imuviso.least_squares_affine_trend(
            t, shifted_midpoint)
        np.testing.assert_allclose(
            scaled + common_line - shifted_trend,
            displayed_scaled,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            integrated + common_line - shifted_trend,
            displayed_integrated,
            atol=1e-14,
        )
        np.testing.assert_array_equal(scaled, scaled_before)
        np.testing.assert_array_equal(integrated, integrated_before)

        initial_coordinate = 3.0 - 0.2 * t + 0.1 * np.sin(0.3 * t)
        coordinate_update = 0.04 * np.cos(1.4 * t)
        final_coordinate = initial_coordinate + coordinate_update
        coordinate_trend = imuviso.least_squares_affine_trend(
            t, initial_coordinate)
        np.testing.assert_allclose(
            (final_coordinate - coordinate_trend)
            - (initial_coordinate - coordinate_trend),
            coordinate_update,
        )

    def test_csv_gravity_is_used_unless_explicitly_overridden(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'flight.csv'
            path.write_text(
                'acc.x;acc.y;vis_x;vis_y;mheight;G\n'
                + '0;0;0;0;10;\n'
                + ''.join('0;0;0;0;10;9.55\n' for _ in range(5)),
                encoding='utf-8')
            flight = imuviso.load_flight_csv(path)

        self.assertEqual(flight.gravity_m_s2, 9.55)
        self.assertEqual(imuviso.resolve_gravity(flight), 9.55)
        self.assertEqual(imuviso.resolve_gravity(flight, 9.7), 9.7)

    def test_positive_coordinate_shift_gives_positive_acceleration_correction(self):
        count = 40
        gravity = 9.5578
        height = np.full(count, 15.0)
        tilt_ratio = 0.02
        raw_acc = np.full(count, -tilt_ratio * gravity)
        vision = np.zeros(count)

        (_, _, coordinate, physical_acc, _, coordinate_delta,
         acceleration_delta, _) = imuviso.rebuild_state(
            raw_acc, vision, height, gravity, 1.0, tilt_ratio)

        np.testing.assert_allclose(coordinate_delta, 0.3)
        np.testing.assert_allclose(coordinate, 0.3)
        np.testing.assert_allclose(acceleration_delta, tilt_ratio * gravity)
        np.testing.assert_allclose(physical_acc, 0.0, atol=1e-14)

    def test_every_iteration_preserves_dx_to_acceleration_coupling(self):
        count = 200
        t = np.arange(count) * imuviso.DT
        height = np.linspace(4.0, 20.0, count)
        raw_acc = 0.15 * np.sin(0.7 * t) - 0.03
        vision = 0.2 * np.sin(0.4 * t) + 0.01 * t**2
        gravity = 9.5578

        (_, _, _, _, states) = imuviso.iterative_true_fit(
            raw_acc, vision, height, gravity, 1.0, 0.3, 3, 1)
        initial_coordinate = states[0]['coordinate']
        for state in states:
            expected = ((state['coordinate'] - initial_coordinate) *
                        gravity / height)
            np.testing.assert_allclose(
                state['physical_acc'] - raw_acc, expected, atol=1e-14)
            np.testing.assert_allclose(
                state['corrected_acc'] - state['physical_acc'],
                state['bias_m_s2'], atol=1e-14)

    def test_returned_output_is_requested_final_iteration(self):
        count = 300
        t = np.arange(count) * imuviso.DT
        height = np.full(count, 2.0)
        raw_acc = np.zeros(count)
        vision = np.sin(2.0 * t)

        scale = 0.7
        coordinate, corrected_acc, integrated, summary, states = (
            imuviso.iterative_true_fit(
                raw_acc, vision, height, 9.5578, scale, 0.3, 3, 1))

        self.assertLess(summary['best_iteration'], 3)
        self.assertEqual(summary['selected_iteration'], 3)
        self.assertEqual(summary['selection'], 'requested final iteration')
        np.testing.assert_array_equal(coordinate, states[-1]['coordinate'])
        np.testing.assert_array_equal(corrected_acc, states[-1]['corrected_acc'])
        np.testing.assert_array_equal(integrated, states[-1]['integrated_position'])
        scaled_vision = scale * vision
        np.testing.assert_allclose(states[0]['coordinate'], scaled_vision)
        scaled_difference = integrated - scaled_vision
        expected_scaled_rmse = float(
            np.sqrt(np.mean(scaled_difference**2)))
        expected_scaled_mean_distance = float(
            np.mean(np.abs(scaled_difference)))
        original_difference = integrated - vision
        expected_original_rmse = float(
            np.sqrt(np.mean(original_difference**2)))
        expected_original_mean_distance = float(
            np.mean(np.abs(original_difference)))
        fit_difference = integrated - coordinate
        expected_fit_rmse = float(np.sqrt(np.mean(fit_difference**2)))

        self.assertAlmostEqual(
            states[-1]['scaled_vision_rmse_m'], expected_scaled_rmse)
        self.assertAlmostEqual(states[-1]['rmse_m'], expected_scaled_rmse)
        self.assertAlmostEqual(
            states[-1]['scaled_vision_mean_abs_distance_m'],
            expected_scaled_mean_distance)
        self.assertAlmostEqual(states[-1]['fit_rmse_m'], expected_fit_rmse)
        self.assertAlmostEqual(
            states[-1]['original_vision_rmse_m'], expected_original_rmse)
        self.assertAlmostEqual(
            states[-1]['original_vision_mean_abs_distance_m'],
            expected_original_mean_distance)
        self.assertAlmostEqual(
            summary['last_iteration_scaled_vision_rmse_m'],
            expected_scaled_rmse)
        self.assertAlmostEqual(
            summary['last_iteration_rmse_m'], expected_scaled_rmse)
        self.assertAlmostEqual(
            summary['last_iteration_original_vision_rmse_m'],
            expected_original_rmse)
        self.assertAlmostEqual(
            summary['last_iteration_fit_rmse_m'], expected_fit_rmse)


if __name__ == '__main__':
    unittest.main()
