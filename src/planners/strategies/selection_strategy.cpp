#include "planners/strategies/selection_strategy.hpp"
#include "planners/strategies/random_selection_strategy.hpp"

namespace namo {

std::unique_ptr<SelectionStrategy> SelectionStrategyFactory::create(StrategyType type) {
    switch (type) {
        case StrategyType::RANDOM:
            return std::make_unique<RandomSelectionStrategy>();
        case StrategyType::ML_DIFFUSION:
            // TODO: Implement ML diffusion strategy with ZMQ communication
            throw std::runtime_error("ML Diffusion strategy not yet implemented");
        case StrategyType::REGION_WAVEFRONT:
            // TODO: Implement region-based wavefront strategy
            throw std::runtime_error("Region Wavefront strategy not yet implemented");
        default:
            throw std::runtime_error("Unknown strategy type");
    }
}

std::unique_ptr<SelectionStrategy> SelectionStrategyFactory::create(const std::string& strategy_name) {
    if (strategy_name == "random" || strategy_name == "Random") {
        return create(StrategyType::RANDOM);
    } else if (strategy_name == "ml_diffusion" || strategy_name == "diffusion") {
        return create(StrategyType::ML_DIFFUSION);
    } else if (strategy_name == "region_wavefront" || strategy_name == "wavefront") {
        return create(StrategyType::REGION_WAVEFRONT);
    } else {
        throw std::runtime_error("Unknown strategy name: " + strategy_name);
    }
}

} // namespace namo