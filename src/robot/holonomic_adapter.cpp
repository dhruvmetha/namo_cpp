#include "robot/holonomic_adapter.hpp"

namespace namo {

HolonomicAdapter::HolonomicAdapter(const std::array<double, 3>& init_pos)
    : init_pos_(init_pos) {}

std::array<double, 2> HolonomicAdapter::get_xy(const mjModel* m, const mjData* d) const {
    // qpos[0:2] are displacements from body origin. Add init_pos to get world frame.
    double x = (m->nq >= 1) ? d->qpos[0] + init_pos_[0] : init_pos_[0];
    double y = (m->nq >= 2) ? d->qpos[1] + init_pos_[1] : init_pos_[1];
    return {x, y};
}

void HolonomicAdapter::set_xy(const mjModel* m, mjData* d, double x, double y) const {
    // Convert world coordinates to qpos displacement
    if (m->nq >= 2) {
        d->qpos[0] = x - init_pos_[0];
        d->qpos[1] = y - init_pos_[1];
        mj_forward(const_cast<mjModel*>(m), d);
    }
}

void HolonomicAdapter::set_se2(const mjModel* m, mjData* d,
                                double x, double y, double /*theta*/) const {
    // Point robot has no heading — ignore theta
    set_xy(m, d, x, y);
}

void HolonomicAdapter::apply_control(const mjModel* m, mjData* d,
                                      double vx, double vy) const {
    if (m->nu >= 2) {
        d->ctrl[0] = vx;
        d->ctrl[1] = vy;
    }
}

void HolonomicAdapter::zero_control(const mjModel* m, mjData* d) const {
    for (int i = 0; i < m->nu; i++) {
        d->ctrl[i] = 0.0;
    }
}

} // namespace namo
