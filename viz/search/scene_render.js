"use strict";
// Scene drawing shared by the episode replay (episode.html) and the scene gallery (scenes.html).
// Everything below is pure: world-frame meters in, SVG string out. The two pages differ only in how
// they colour the contact points, so that part stays with each page.

// Walk the row-wise RLE back into one rectangle per run (see trace_schema.rle_encode).
function regionRuns(regions) {
  const { nx, ny, res, origin, rle } = regions;
  const runs = [];
  let ix = 0;
  let iy = 0;
  for (let i = 0; i < rle.length && ix < nx; i += 2) {
    const v = rle[i];
    let n = rle[i + 1];
    while (n > 0 && ix < nx) {
      const take = Math.min(n, ny - iy);
      if (v !== 0) runs.push({ v, x: origin[0] + ix * res, y: origin[1] + iy * res, h: take * res });
      iy += take;
      n -= take;
      if (iy >= ny) {
        iy = 0;
        ix += 1;
      }
    }
  }
  return runs;
}

// The problem in one picture: "robot" = where the robot can currently get to, "goal" = the pocket it
// is trying to reach, and a push succeeds exactly when the two become one region ("robot_goal").
// Everything else is background free space -- drawn, but deliberately dull.
function regionClass(label) {
  if (label === "robot") return "region-robot";
  if (label === "goal") return "region-goal";
  if (label === "robot_goal") return "region-merged";
  return "region-other";
}

function regionLayer(regions) {
  if (!regions) return "";
  const res = regions.res;
  const cells = regionRuns(regions).map((r) => {
    const label = regions.labels[String(r.v)] || `region_${r.v}`;
    // 2% overhang on the width so neighbouring columns of the same region overlap instead of
    // leaving antialiased hairlines between them (0.1 mm of overdraw at 5 mm cells).
    return (
      `<rect class="region-cell ${regionClass(label)}" x="${r.x}" y="${r.y}" ` +
      `width="${res * 1.02}" height="${r.h}"><title>${label}</title></rect>`
    );
  });
  return `<g class="region-layer">${cells.join("")}</g>`;
}

// Set the SVG frame from the scene bounds. No Y-flip: contacts arrive as raw world (x,y) already
// rotated by contact_offsets_world's standard [cos -sin; sin cos] matrix, and the rect rotate()
// below uses the same matrix on the same raw coordinates, so leaving the frame as-is keeps contact
// points and rectangle edges aligned. (Screen "up" ends up meaning +y is drawn toward larger SVG y,
// i.e. lower on screen -- a mirrored-but-internally-consistent convention, harmless for a
// diagnostic tool with no photo to match against.)
// `fit` (gallery only) also sizes the ELEMENT to the room's proportions. Without it a portrait room
// in a landscape box is letterboxed by preserveAspectRatio, which reads as a broken layout rather
// than as a tall room. The replay page keeps its fixed square box, where the geometry changes as the
// search moves objects and a resizing panel would be worse.
function setSceneViewBox(svg, scene, fit) {
  const [xmin, xmax, ymin, ymax] = scene.bounds;
  const w = xmax - xmin, h = ymax - ymin;
  svg.setAttribute("viewBox", `${xmin} ${ymin} ${w} ${h}`);
  if (fit) {
    svg.style.aspectRatio = `${w} / ${h}`;
    // leave room for the filter bar so the whole room is visible without scrolling
    svg.style.height = "min(calc(100vh - 190px), 900px)";
    svg.style.width = "auto";
    svg.style.maxWidth = "100%";
  }
}

// Everything except the contact points: region tint, walls, goal marker, movable boxes, robot.
// `geom` (optional) overrides the poses for a mid-search state; without it the scene's own start
// poses are drawn. `targetId` gets the highlighted class.
function sceneLayers(scene, geom, regions, targetId) {
  const parts = [];
  const stroke = 0.0025;

  parts.push(regionLayer(regions)); // first = beneath everything else

  for (const s of scene.static) {
    const deg = (2 * Math.atan2(s.qz, s.qw) * 180) / Math.PI;
    parts.push(
      `<rect x="${-s.hw}" y="${-s.hd}" width="${2 * s.hw}" height="${2 * s.hd}" class="wall-rect" ` +
        `transform="translate(${s.x},${s.y}) rotate(${deg})"/>`
    );
  }

  const [gx, gy] = scene.goal;
  const gr = 0.02;
  parts.push(
    `<g class="goal-marker" transform="translate(${gx},${gy})">` +
      `<line x1="${-gr}" y1="${-gr}" x2="${gr}" y2="${gr}" stroke-width="${stroke * 2}"/>` +
      `<line x1="${-gr}" y1="${gr}" x2="${gr}" y2="${-gr}" stroke-width="${stroke * 2}"/>` +
      `<circle r="${gr * 1.3}" class="goal-ring"/></g>`
  );

  for (const m of scene.movable) {
    const [mx, my, mtheta] = (geom && geom.movable[m.name]) || [m.x, m.y, m.theta];
    const deg = (mtheta * 180) / Math.PI;
    parts.push(
      `<rect x="${-m.hw}" y="${-m.hd}" width="${2 * m.hw}" height="${2 * m.hd}" ` +
        `class="${m.name === targetId ? "movable-target" : "movable-other"}" ` +
        `transform="translate(${mx},${my}) rotate(${deg})"><title>${m.name}</title></rect>`
    );
  }

  const [rx, ry, rtheta] = (geom && geom.robot) || scene.robot;
  const rr = 0.025;
  parts.push(
    `<g class="robot-marker" transform="translate(${rx},${ry}) rotate(${(rtheta * 180) / Math.PI})">` +
      `<circle r="${rr}"/><line x1="0" y1="0" x2="${rr * 1.6}" y2="0" stroke-width="${stroke * 2}"/></g>`
  );

  return parts;
}
