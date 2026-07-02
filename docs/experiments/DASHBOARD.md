# 🧭 Research Dashboard

> **Obsidian-only view.** These tables auto-build from each note's `status` frontmatter via the **Dataview**
> community plugin (Settings → Community plugins → install & enable *Dataview*). Without the plugin the blocks
> below show as raw code — that's expected; on GitHub or in Claude, use **[INDEX.md](../INDEX.md)** instead
> (the plain, always-valid index). Nothing here needs maintaining: add/point a note's `status:` and it moves itself.

## 🟢 Live — active threads
```dataview
TABLE WITHOUT ID file.link AS "Note", updated AS "Updated"
FROM #experiment
WHERE status = "live"
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

## 📸 Snapshots — point-in-time results (may be stale)
```dataview
TABLE WITHOUT ID file.link AS "Note", updated AS "Updated"
FROM #experiment
WHERE status = "snapshot"
SORT updated DESC
```

## ❄️ Frozen — evidence archive, do not overwrite
```dataview
TABLE WITHOUT ID file.link AS "Note", updated AS "Updated"
FROM #experiment
WHERE status = "frozen"
SORT file.name ASC
```

---
*Statuses: `live` active · `hub` catalog · `ref` stable brief · `snapshot` dated result · `frozen` archive.
Set them in each note's frontmatter. This dashboard + [INDEX.md](../INDEX.md) are two front ends onto the same files.*
