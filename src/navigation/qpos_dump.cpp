#include "navigation/qpos_dump.hpp"
#include "environment/namo_environment.hpp"

#include <cstdio>
#include <cstdlib>

namespace namo {

void dump_qpos(NAMOEnvironment& env, int phase_id) {
    static FILE* fp = nullptr;
    static bool init = false;
    if (!init) {
        const char* path = std::getenv("NAMO_QPOS_DUMP");
        if (path && path[0]) fp = std::fopen(path, "w");
        init = true;
    }
    if (!fp) return;

    auto* m = env.get_mujoco_wrapper()->model();
    auto* d = env.get_mujoco_wrapper()->data();
    std::fprintf(fp, "%d %d", phase_id, m->nq);
    for (int i = 0; i < m->nq; i++) std::fprintf(fp, " %.6f", d->qpos[i]);
    std::fprintf(fp, "\n");
    std::fflush(fp);
}

} // namespace namo
