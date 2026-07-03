# 🧭 Research Dashboard

> **Obsidian-only view.** The experiment board is native **Bases** — open **[experiments.base](experiments.base)**
> (enable core Bases; then "Group by → status"). The tables below use **Dataview** (community plugin) as a
> fallback and for the journal/reference docs. Without either plugin the blocks show as raw code — on
> GitHub / in Claude use **[INDEX.md](../INDEX.md)** (plain map) and **[RESULTS.md](RESULTS.md)** (compiled results).

## 🧪 Experiments — the loop (`idea → live → done`)
Open the board: **[experiments.base](experiments.base)** (grouped by `status`). Primary view for active
experiment cards. Compiled results: [RESULTS.md](RESULTS.md).

## 🟢 Live docs — active journals & ledgers
```dataview
TABLE WITHOUT ID file.link AS "Note", updated AS "Updated"
FROM #experiment
WHERE status = "live" AND type != "experiment"
SORT updated DESC
```

## 🎯 Hub — start-here catalogs (read for paths, never glob)
```dataview
TABLE WITHOUT ID file.link AS "Note", updated AS "Updated"
FROM #experiment
WHERE status = "hub"
SORT file.name ASC
```

## 📎 Reference — stable briefs & positioning
```dataview
TABLE WITHOUT ID file.link AS "Note", updated AS "Updated"
FROM #experiment
WHERE status = "ref"
SORT file.name ASC
```

---
*Statuses — experiments: `idea → live → done`; docs: `live` active · `hub` catalog · `ref` brief.
Compiled results: [RESULTS.md](RESULTS.md) · how the loop works: [WORKFLOW.md](WORKFLOW.md) · plain map: [INDEX.md](../INDEX.md).*
