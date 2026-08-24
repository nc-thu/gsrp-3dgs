import unittest

from gs_arch_sim_v11.io import validate_tile
from gs_arch_sim_v11.model import Spec, replay_camera, schedule, tile_cycles


def tile(tile_id=0, raw=10, retained=6, producer_raw=40, producer_retained=20):
    return {
        "tile_id": tile_id, "n_kvp_raw": raw, "n_kvp_retained": retained,
        "n_kvp_tstopped": raw - retained, "producer_cycles_raw": producer_raw,
        "producer_cycles_retained": producer_retained, "tstop_after_kvp": retained,
    }


class ModelTests(unittest.TestCase):
    def test_tile_formula(self):
        spec = Spec(fill_drain_cycles=30)
        self.assertEqual(tile_cycles(tile(), False, spec), 190)
        self.assertEqual(tile_cycles(tile(), True, spec), 126)

    def test_t_only_removes_future_kvp(self):
        row = tile(raw=10, retained=1)
        validate_tile(row)
        self.assertEqual(row["n_kvp_tstopped"], 9)

    def test_earliest_free_tie_break(self):
        rows = [tile(index, raw=1, retained=1, producer_raw=1, producer_retained=1) for index in range(5)]
        cycles, loads = schedule(rows, False, Spec(fill_drain_cycles=0))
        self.assertEqual(cycles, 32)
        self.assertEqual(loads, [32, 16, 16, 16])

    def test_replay_closure(self):
        result = replay_camera([tile()])
        self.assertEqual(result["n_kvp_raw"], 10)
        self.assertEqual(result["n_kvp_retained"] + result["n_kvp_tstopped"], 10)
        self.assertGreater(result["t_speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
