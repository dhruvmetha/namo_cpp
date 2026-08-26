"""Generate scratch XML variants. Never edits the source scene in place."""
import os, re, shutil
SRC="/common/users/dm1487/scratch_namo/real_buildable/pool1/hard/rb_00363/env.xml"
OUT="/tmp/claude-89862/-common-home-dm1487-robotics-research-ktamp-namo/a699fcde-7931-4d4f-a2d6-9dc120966a3b/scratchpad/rot/xml"

BLOCK_RE = re.compile(r'(<geom name="obstacle_0_movable"[^>]*?)/>')
FLOOR_RE = re.compile(r'(<geom name="floor"[^>]*?)/>')
CHASSIS_RE = re.compile(r'(<geom name="(?:front|rear)_chassis_collision"[^>]*?)/>')
OPT_RE = re.compile(r'<option ([^>]*)/>')

def _set_attr(tag, k, v):
    if re.search(rf'{k}="[^"]*"', tag):
        return re.sub(rf'{k}="[^"]*"', f'{k}="{v}"', tag)
    return tag + f' {k}="{v}"'

def make(name, block=None, floor=None, chassis=None, option=None):
    s = open(SRC).read()
    if block:
        s = BLOCK_RE.sub(lambda m: _apply(m.group(1), block) + "/>", s)
    if floor:
        s = FLOOR_RE.sub(lambda m: _apply(m.group(1), floor) + "/>", s)
    if chassis:
        s = CHASSIS_RE.sub(lambda m: _apply(m.group(1), chassis) + "/>", s)
    if option:
        s = OPT_RE.sub(lambda m: "<option " + _apply(m.group(1), option) + "/>", s)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.xml")
    open(p, "w").write(s)
    return p

def _apply(tag, kv):
    for k, v in kv.items():
        tag = _set_attr(tag, k, v)
    return tag

VARIANTS = {
  "base":            {},
  # --- torsional friction on the block (peer's top suspect) ---
  "tors_0001":       dict(block={"friction":"1 0.0001 0.0001"}),
  "tors_005":        dict(block={"friction":"1 0.05 0.0001"}),
  "tors_05":         dict(block={"friction":"1 0.5 0.0001"}),
  "tors_5":          dict(block={"friction":"1 5.0 0.0001"}),
  # --- block sliding friction (block priority wins over floor AND car) ---
  "blockmu_06":      dict(block={"friction":"0.6 0.005 0.0001","priority":"1"}),
  "blockmu_03":      dict(block={"friction":"0.3 0.005 0.0001","priority":"1"}),
  "blockmu_015":     dict(block={"friction":"0.15 0.005 0.0001","priority":"1"}),
  # --- floor only (block-floor + wheel traction) ---
  "floormu_03":      dict(floor={"friction":"0.3 0.005 0.001","priority":"1"}),
  "floormu_015":     dict(floor={"friction":"0.15 0.005 0.001","priority":"1"}),
  # --- pusher interface only (chassis priority; wheels untouched) ---
  "pushmu_03":       dict(chassis={"friction":"0.3 0.005 0.0001","priority":"2"}),
  "pushmu_01":       dict(chassis={"friction":"0.1 0.005 0.0001","priority":"2"}),
  # --- mass ---
  "mass_005":        dict(block={"mass":"0.05"}),
  "mass_02":         dict(block={"mass":"0.2"}),
  "mass_05":         dict(block={"mass":"0.5"}),
  # --- contact stiffness / solver ---
  "solref_stiff":    dict(block={"solref":"0.004 1"}),
  "solref_vstiff":   dict(block={"solref":"0.002 1"}),
  "solref_soft":     dict(block={"solref":"0.05 1"}),
  "solimp_hard":     dict(block={"solimp":"0.99 0.9999 0.0001 0.5 2"}),
  "impratio_10":     dict(option={"impratio":"10"}),
  "cone_pyr":        dict(option={"cone":"pyramidal"}),
  "condim3":         dict(block={"condim":"3"}),
  "condim6":         dict(block={"condim":"6"}),
  # --- combinations ---
  "mu03_stiff":      dict(block={"friction":"0.3 0.005 0.0001","priority":"1","solref":"0.004 1"}),
  "mu015_stiff":     dict(block={"friction":"0.15 0.005 0.0001","priority":"1","solref":"0.004 1"}),
  "floor03_push03":  dict(floor={"friction":"0.3 0.005 0.001","priority":"1"},
                          chassis={"friction":"0.3 0.005 0.0001","priority":"2"}),
}


# Isolating variants: block-floor friction ONLY.
# chassis priority 2 keeps the pusher grip at 1.0; wheels stay priority 0 so
# wheel-floor traction is untouched at max(1,0.5)=1.0.
_HIGH_CHASSIS = {"friction":"1 0.005 0.0001","priority":"2"}
for _mu in ("0.6","0.3","0.15","0.05"):
    VARIANTS[f"iso_floor_{_mu.replace('.','')}"] = dict(
        block={"friction":f"{_mu} 0.005 0.0001","priority":"1"}, chassis=dict(_HIGH_CHASSIS))
# isolated block-floor friction + stiffer contact
for _mu in ("0.3","0.15"):
    VARIANTS[f"iso_stiff_{_mu.replace('.','')}"] = dict(
        block={"friction":f"{_mu} 0.005 0.0001","priority":"1","solref":"0.004 1"},
        chassis=dict(_HIGH_CHASSIS))
# isolated, plus raised torsional (does torsion help ONCE the block can spin?)
VARIANTS["iso_03_tors05"] = dict(
    block={"friction":"0.3 0.5 0.0001","priority":"1"}, chassis=dict(_HIGH_CHASSIS))

if __name__=="__main__":
    for n,kw in VARIANTS.items():
        print(make(n, **kw))
