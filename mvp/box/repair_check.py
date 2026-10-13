"""Exercise mask+weights repair on a mask with removed sub-blocks (shakedown of the optional mode)."""
import json, sys
from chantless.pipeline import Run
from chantless import mask as mk
from chantless.util import read_json
cfg = json.load(open(sys.argv[1])); cfg["run_id"] = sys.argv[2]
R = Run(cfg); R.guard.phase = "search"
m = read_json(R.dir / "sanity" / "sanity_mask.json")
floors = read_json(R.dir / "sanity" / "floors.json")
drops = [g for g, f in floors.items() if f["drop_ok"]][:2]
for g in drops:
    m = mk.apply(m, g, "drop")
print("drops", drops, flush=True)
P = R.make_prober(R.src, "repaircheck")
w, lineage, m2 = R._repair(m, P)
print(json.dumps(lineage, indent=1)[:1500])
