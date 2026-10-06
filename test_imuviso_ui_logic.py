import unittest

from imuviso_ui import (
    _EndpointPoint,
    _history_point_from_result,
    _store_endpoint_point,
)


def _point(tick, value):
    return _EndpointPoint(
        tick=tick,
        scaled_x=value,
        integrated_x=value + 0.1,
        scaled_y=-value,
        integrated_y=-value - 0.1,
        initial_bias_x=value + 1.0,
        final_bias_x=value + 2.0,
        initial_bias_y=-value - 1.0,
        final_bias_y=-value - 2.0,
    )


class RealtimeHistoryTests(unittest.TestCase):
    def test_history_storage_keeps_biases_aligned_with_endpoints(self):
        history = []
        _store_endpoint_point(history, _point(100, 1.0), 3)
        replacement = _point(100, 2.0)
        _store_endpoint_point(history, replacement, 3)

        self.assertEqual(history, [replacement])

        _store_endpoint_point(history, _point(110, 3.0), 3)
        self.assertEqual([item.tick for item in history], [100, 110])

        _store_endpoint_point(history, _point(111, 4.0), 3)
        _store_endpoint_point(history, _point(112, 5.0), 3)
        self.assertEqual([item.tick for item in history], [110, 111, 112])
        self.assertEqual(history[-1].final_bias_x, 7.0)

        _store_endpoint_point(history, _point(123, 6.0), 3)
        self.assertEqual([item.tick for item in history], [123])

        _store_endpoint_point(history, _point(120, 7.0), 3)
        self.assertEqual([item.tick for item in history], [120])

    def test_result_extracts_iteration_zero_and_requested_final_bias(self):
        result = {
            "start": 20,
            "count": 5,
            "axes": {
                "x": {
                    "scaled_vision": [10.0, 11.0],
                    "states": [
                        {"bias_m_s2": 1.0, "integrated_position": [20.0]},
                        {"bias_m_s2": 99.0, "integrated_position": [21.0]},
                        {"bias_m_s2": 3.0, "integrated_position": [22.0]},
                    ],
                    "summary": {"best_bias_m_s2": 99.0},
                },
                "y": {
                    "scaled_vision": [-10.0, -11.0],
                    "states": [
                        {"bias_m_s2": -1.0, "integrated_position": [-20.0]},
                        {"bias_m_s2": -99.0, "integrated_position": [-21.0]},
                        {"bias_m_s2": -3.0, "integrated_position": [-22.0]},
                    ],
                    "summary": {"best_bias_m_s2": -99.0},
                },
            },
        }

        point = _history_point_from_result(result)

        self.assertEqual(point.tick, 24)
        self.assertEqual((point.scaled_x, point.integrated_x), (11.0, 22.0))
        self.assertEqual((point.scaled_y, point.integrated_y), (-11.0, -22.0))
        self.assertEqual((point.initial_bias_x, point.final_bias_x), (1.0, 3.0))
        self.assertEqual((point.initial_bias_y, point.final_bias_y), (-1.0, -3.0))


if __name__ == "__main__":
    unittest.main()
