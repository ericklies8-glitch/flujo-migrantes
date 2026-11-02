import unittest
from backend.scenario_model import (
    CONFIG_ESCENARIOS, CAPACIDADES_NODOS, CAPACIDAD_RED_CORREDOR,
    STOCK_INICIAL_RETENIDOS, simular_determinista, simular_montecarlo,
)

class TestScenarioModel(unittest.TestCase):
    def test_capacity(self):
        self.assertEqual(CAPACIDAD_RED_CORREDOR, sum(CAPACIDADES_NODOS.values()))
        self.assertEqual(CAPACIDAD_RED_CORREDOR, 19000)

    def test_rate_complement(self):
        for p in CONFIG_ESCENARIOS.values():
            a = p["tasa_integracion"]
            self.assertTrue(0 <= a <= 1)
            self.assertAlmostEqual(a + (1 - a), 1)

    def test_deterministic_reproducibility(self):
        a = simular_determinista("Línea Base", 12)
        self.assertEqual(a, simular_determinista("Línea Base", 12))
        self.assertEqual(len(a["serie_temporal"]), 12)

    def test_saturation_unclipped(self):
        result = simular_determinista("Crisis +50%", 12)
        self.assertGreater(result["serie_temporal"][-1]["saturacion_corredor_pct"], 100)

    def test_seed_reproducibility(self):
        a = simular_montecarlo("Crisis +50%", 12345, 12, 1)
        b = simular_montecarlo("Crisis +50%", 12345, 12, 1)
        self.assertEqual(a, b)
        self.assertEqual(len(a[1]), 12)

    def test_old_threshold_invalid(self):
        self.assertGreater(STOCK_INICIAL_RETENIDOS, 250)

if __name__ == "__main__":
    unittest.main()
