#include "planners/strategies/strategy_factory.hpp"
#include "planners/strategies/random_selection_strategy.hpp"
#include "planners/strategies/ml_diffusion_strategy.hpp"
#include <stdexcept>
#include <algorithm>
#include <mutex>

namespace namo {

// Static member definitions
std::map<StrategyFactory::Type, StrategyFactory::StrategyCreator> StrategyFactory::strategy_creators_;
std::map<std::string, StrategyFactory::Type> StrategyFactory::name_to_type_;
bool StrategyFactory::initialized_ = false;

void StrategyFactory::initialize() {
    if (initialized_) return;
    
    // Register built-in strategies
    strategy_creators_[Type::RANDOM] = [](std::shared_ptr<ConfigManager> config) {
        return std::make_unique<RandomSelectionStrategy>(config);
    };
    
    strategy_creators_[Type::ML_DIFFUSION] = [](std::shared_ptr<ConfigManager> config) {
        return std::make_unique<MLDiffusionStrategy>(
            config->strategy().zmq_endpoint,
            config->strategy().zmq_timeout_ms,
            config->strategy().fallback_to_random
        );
    };
    
    // TODO: Add region wavefront strategy when implemented
    // strategy_creators_[Type::REGION_WAVEFRONT] = [](std::shared_ptr<ConfigManager> config) {
    //     return std::make_unique<RegionWavefrontStrategy>(config);
    // };
    
    // Register name mappings (case-insensitive)
    name_to_type_["random"] = Type::RANDOM;
    name_to_type_["Random"] = Type::RANDOM;
    name_to_type_["RANDOM"] = Type::RANDOM;
    
    name_to_type_["ml_diffusion"] = Type::ML_DIFFUSION;
    name_to_type_["ML_Diffusion"] = Type::ML_DIFFUSION;
    name_to_type_["diffusion"] = Type::ML_DIFFUSION;
    name_to_type_["ml"] = Type::ML_DIFFUSION;
    
    name_to_type_["region_wavefront"] = Type::REGION_WAVEFRONT;
    name_to_type_["wavefront"] = Type::REGION_WAVEFRONT;
    name_to_type_["region"] = Type::REGION_WAVEFRONT;
    
    initialized_ = true;
}

void StrategyFactory::ensure_initialized() {
    static std::once_flag init_flag;
    std::call_once(init_flag, initialize);
}

std::unique_ptr<SelectionStrategy> StrategyFactory::create(Type type, 
                                                         std::shared_ptr<ConfigManager> config) {
    ensure_initialized();
    
    if (!config) {
        throw std::invalid_argument("Configuration manager cannot be null");
    }
    
    auto it = strategy_creators_.find(type);
    if (it == strategy_creators_.end()) {
        throw std::invalid_argument("Unknown strategy type: " + std::to_string(static_cast<int>(type)));
    }
    
    try {
        auto strategy = it->second(config);
        if (!strategy) {
            throw std::runtime_error("Strategy creator returned null pointer");
        }
        return strategy;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create strategy: " + std::string(e.what()));
    }
}

std::unique_ptr<SelectionStrategy> StrategyFactory::create(const std::string& name,
                                                         std::shared_ptr<ConfigManager> config) {
    ensure_initialized();
    
    if (!config) {
        throw std::invalid_argument("Configuration manager cannot be null");
    }
    
    auto it = name_to_type_.find(name);
    if (it == name_to_type_.end()) {
        throw std::invalid_argument("Unknown strategy name: " + name);
    }
    
    return create(it->second, config);
}

std::unique_ptr<SelectionStrategy> StrategyFactory::create_with_defaults(Type type) {
    auto default_config = ConfigManager::create_default();
    return create(type, std::shared_ptr<ConfigManager>(default_config.release()));
}

void StrategyFactory::register_strategy(const std::string& name, StrategyCreator creator) {
    ensure_initialized();
    
    if (name.empty()) {
        throw std::invalid_argument("Strategy name cannot be empty");
    }
    
    if (!creator) {
        throw std::invalid_argument("Strategy creator cannot be null");
    }
    
    // Register as custom strategy
    strategy_creators_[Type::CUSTOM] = creator;
    name_to_type_[name] = Type::CUSTOM;
}

std::vector<std::string> StrategyFactory::get_available_strategies() {
    ensure_initialized();
    
    std::vector<std::string> names;
    for (const auto& [name, type] : name_to_type_) {
        names.push_back(name);
    }
    
    std::sort(names.begin(), names.end());
    return names;
}

bool StrategyFactory::validate_config(Type type, const ConfigManager& config) {
    ensure_initialized();
    
    try {
        switch (type) {
            case Type::RANDOM:
                // Validate random strategy configuration
                if (config.strategy().min_goal_distance <= 0 ||
                    config.strategy().max_goal_distance <= config.strategy().min_goal_distance ||
                    config.strategy().max_goal_attempts <= 0 ||
                    config.strategy().max_object_retries <= 0) {
                    return false;
                }
                break;
                
            case Type::ML_DIFFUSION:
                // Validate ML strategy configuration
                if (config.strategy().zmq_endpoint.empty() ||
                    config.strategy().zmq_timeout_ms <= 0) {
                    return false;
                }
                break;
                
            case Type::REGION_WAVEFRONT:
                // TODO: Add validation when implemented
                break;
                
            case Type::CUSTOM:
                // Cannot validate custom strategies without additional information
                break;
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace namo