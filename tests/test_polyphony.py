"""Small exhaustive oracles for the occurrence-level polyphony protocol."""
import unittest
from itertools import product

import numpy as np

from phonetics.research.polyphony import direction_score, plan_polyphony, evaluate_polyphony


class PolyphonyTests(unittest.TestCase):
    def test_direction_matches_exhaustive_with_ties_and_unequal_support(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(7, 5)), rng.normal(size=(3, 5))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y[0] = x[1]
        for tolerance in (0, 1e-12, .15):
            result = direction_score(x, y, tolerance)
            scores, gaps, within, cross = [], [], [], []
            for i,j,k in product(range(len(x)), range(len(x)), range(len(y))):
                if i == j:
                    continue
                w, c = np.clip(1-x[i]@x[j],0,2), np.clip(1-x[i]@y[k],0,2)
                gap = c-w
                scores.append(1 if gap > tolerance else 0 if gap < -tolerance else .5)
                gaps.append(gap); within.append(w); cross.append(c)
            for key, values in (("score",scores),("delta",gaps),("within_distance",within),("cross_distance",cross)):
                self.assertAlmostEqual(result[key], np.mean(values))

    def test_all_ties(self):
        r = direction_score(np.ones((3,1)), np.ones((2,1)), 1e-12)
        self.assertEqual((r["score"], r["tie_rate"], r["delta"]), (.5, 1, 0))

    def fixture(self):
        rows = []
        for char, reading, count in (("X","A",3),("X","B",4),("X","C",2),
            ("Y","A",2),("Y","B",2),("Z","B",8)):
            rows += [{"character":char,"pinyin":reading,"occurrence_id":str(len(rows)+i)} for i in range(count)]
        return rows

    def test_three_readings_seed_intersection_and_control_support(self):
        rows = self.fixture()
        full = list(range(len(rows)))
        partial = [i for i in full if not (rows[i]["character"]=="X" and rows[i]["pinyin"]=="C")]
        _, plan, coverage = plan_polyphony(rows,{0:full,1:partial})
        self.assertEqual([(r["character"],r["reading_a"],r["reading_b"]) for r in plan],[("X","A","B"),("Y","A","B")])
        self.assertTrue(plan[0]["matched_control"])
        self.assertEqual(plan[0]["controls"],{"A":["Y"],"B":["Y","Z"]})
        self.assertTrue(any(r["reason"] != "included" for r in coverage))

    def test_equal_pairs_and_control_characters_and_empty_high_slice(self):
        rows = self.fixture()
        groups, plan, _ = plan_polyphony(rows,{0:list(range(len(rows)))})
        self.assertEqual(len(plan),4)
        self.assertEqual(sum(r["matched_control"] for r in plan),2)
        hidden=np.random.default_rng(2).normal(size=(len(rows),6))
        tables=evaluate_polyphony(hidden,groups[0],plan,1e-12,"encoder")
        summary={ (r["slice"],r["population"]):r for r in tables["polyphony_summary"]}
        pairs=[r for r in tables["polyphony_pairs"] if r["population"]=="main"]
        self.assertAlmostEqual(summary["main","main"]["score"],np.mean([
            np.mean([r["score"] for r in pairs if r["character"]==char]) for char in ("X","Y")]))
        self.assertIsNone(summary["high_support","main"]["score"])
        direction=next(r for r in tables["polyphony_directions"] if r["direction"]=="A -> B" and r["population"]=="control")
        norm=hidden/np.linalg.norm(hidden,axis=1,keepdims=True)
        oracle=np.mean([direction_score(norm[groups[0]["X","A"]],norm[groups[0][y,"B"]],1e-12)["score"] for y in ("Y","Z")])
        self.assertAlmostEqual(direction["score"],oracle)
        self.assertEqual(direction["wins"]+direction["ties"]+direction["losses"],direction["triples"])

    def test_high_support_threshold_and_one_sided_control(self):
        rows = []
        for char, reading, n in (("X","A",11),("X","B",11),("Y","A",11),("Z","B",10)):
            rows.extend({"character":char,"pinyin":reading} for _ in range(n))
        _,plan,_=plan_polyphony(rows,{0:list(range(len(rows))),1:list(range(len(rows)))})
        high=next(r for r in plan if r["slice"]=="high_support")
        self.assertFalse(high["matched_control"])
        self.assertEqual(high["controls"],{"A":["Y"],"B":[]})
        _,filtered,_=plan_polyphony(rows,{0:list(range(len(rows))),1:list(range(1,len(rows)))})
        self.assertFalse(any(r["slice"]=="high_support" for r in filtered))


if __name__ == "__main__":
    unittest.main()
