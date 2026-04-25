#pragma once

namespace namo {
class NAMOEnvironment;

/// Dump full qpos to the file named by the NAMO_QPOS_DUMP env var (if set).
/// No-op if the env var is unset. The file is opened on first call and held
/// open for the process lifetime; repeated calls append a frame per call.
/// `phase_id` is written as the first token on each line for downstream
/// phase-coloring.
void dump_qpos(NAMOEnvironment& env, int phase_id);

} // namespace namo
