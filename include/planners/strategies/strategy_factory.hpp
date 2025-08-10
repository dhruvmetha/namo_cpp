#pragma once

#include "planners/strategies/selection_strategy.hpp"
#include "config/config_manager.hpp"
#include <memory>
#include <string>
#include <map>
#include <functional>

namespace namo {

/**
 * @brief Robust factory for creating selection strategies with proper configuration injection
 * 
 * Thread-safe factory that manages strategy creation with consistent configuration
 * management and error handling.
 */
class StrategyFactory {
public:
    enum class Type {
        RANDOM,
        ML_DIFFUSION,
        REGION_WAVEFRONT,
        CUSTOM
    };
    
    using StrategyCreator = std::function<std::unique_ptr<SelectionStrategy>(std::shared_ptr<ConfigManager>)>;
    
private:
    static std::map<Type, StrategyCreator> strategy_creators_;
    static std::map<std::string, Type> name_to_type_;
    static bool initialized_;
    
    static void initialize();
    static void ensure_initialized();

public:
    /**
     * @brief Create strategy by type with configuration
     * @param type Strategy type
     * @param config Configuration manager (required)
     * @return Unique pointer to strategy instance
     * @throws std::invalid_argument if type unknown or config invalid
     */
    static std::unique_ptr<SelectionStrategy> create(Type type, 
                                                   std::shared_ptr<ConfigManager> config);
    
    /**
     * @brief Create strategy by name with configuration
     * @param name Strategy name (case-insensitive)
     * @param config Configuration manager (required)
     * @return Unique pointer to strategy instance
     * @throws std::invalid_argument if name unknown or config invalid
     */
    static std::unique_ptr<SelectionStrategy> create(const std::string& name,
                                                   std::shared_ptr<ConfigManager> config);
    
    /**
     * @brief Create strategy with default configuration
     * @param type Strategy type
     * @return Unique pointer to strategy instance with default config
     */
    static std::unique_ptr<SelectionStrategy> create_with_defaults(Type type);
    
    /**
     * @brief Register custom strategy creator
     * @param name Strategy name
     * @param creator Factory function
     */
    static void register_strategy(const std::string& name, StrategyCreator creator);
    
    /**
     * @brief Get list of available strategy names
     * @return Vector of registered strategy names
     */
    static std::vector<std::string> get_available_strategies();
    
    /**
     * @brief Validate strategy configuration
     * @param type Strategy type
     * @param config Configuration to validate
     * @return True if configuration is valid for strategy
     */
    static bool validate_config(Type type, const ConfigManager& config);
};

} // namespace namo