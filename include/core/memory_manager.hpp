#pragma once

#include "core/types.hpp"
#include <array>
#include <memory>
#include <cassert>
#include <iostream>
#include <functional>

namespace namo {

/**
 * @brief Object pool for zero-allocation runtime performance
 * 
 * Pre-allocates a fixed number of objects and manages their lifetime
 * without any dynamic memory allocation during runtime.
 */
template<typename T, size_t POOL_SIZE>
class ObjectPool {
private:
    std::array<T, POOL_SIZE> objects_;
    std::array<bool, POOL_SIZE> available_;
    size_t next_available_ = 0;
    size_t objects_in_use_ = 0;
    
public:
    ObjectPool() {
        available_.fill(true);
    }
    
    /**
     * @brief Acquire an object from the pool
     * @return Pointer to available object, or nullptr if pool exhausted
     */
    T* acquire() {
        // Fast path: check next available slot
        if (available_[next_available_]) {
            available_[next_available_] = false;
            objects_in_use_++;
            T* obj = &objects_[next_available_];
            
            // Find next available slot for next acquisition
            find_next_available();
            return obj;
        }
        
        // Slow path: search for any available slot
        for (size_t i = 0; i < POOL_SIZE; i++) {
            if (available_[i]) {
                available_[i] = false;
                objects_in_use_++;
                next_available_ = i;
                find_next_available();
                return &objects_[i];
            }
        }
        
        // Pool exhausted
        return nullptr;
    }
    
    /**
     * @brief Return an object to the pool
     * @param obj Pointer to object to return
     */
    void release(T* obj) {
        if (!obj) return;
        
        // Calculate index from pointer
        size_t index = obj - objects_.data();
        assert(index < POOL_SIZE);
        assert(!available_[index]); // Object should be in use
        
        // Reset object state
        obj->clear();
        
        // Mark as available
        available_[index] = true;
        objects_in_use_--;
        
        // Update next available if this slot is better
        if (index < next_available_) {
            next_available_ = index;
        }
    }
    
    /**
     * @brief Reset all objects to available state
     */
    void reset_all() {
        available_.fill(true);
        next_available_ = 0;
        objects_in_use_ = 0;
        
        // Clear all objects
        for (auto& obj : objects_) {
            obj.clear();
        }
    }
    
    /**
     * @brief Get pool statistics
     */
    size_t size() const { return POOL_SIZE; }
    size_t available_count() const { return POOL_SIZE - objects_in_use_; }
    size_t used_count() const { return objects_in_use_; }
    bool is_exhausted() const { return objects_in_use_ == POOL_SIZE; }
    
private:
    void find_next_available() {
        for (size_t i = next_available_ + 1; i < POOL_SIZE; i++) {
            if (available_[i]) {
                next_available_ = i;
                return;
            }
        }
        // If not found, wrap around
        for (size_t i = 0; i < next_available_; i++) {
            if (available_[i]) {
                next_available_ = i;
                return;
            }
        }
        next_available_ = 0; // Will be checked in acquire()
    }
};

/**
 * @brief Centralized memory manager for NAMO system
 * 
 * Pre-allocates all memory pools used throughout the system
 * to ensure zero runtime allocations.
 */
class NAMOMemoryManager {
private:
    // Object pools for different types
    ObjectPool<State, 1000> state_pool_;
    ObjectPool<Control, 500> control_pool_;
    ObjectPool<ActionStepMPC, 200> action_pool_;
    ObjectPool<MotionPrimitive, 2000> primitive_pool_;
    ObjectPool<GridFootprint, 100> footprint_pool_;
    
    // Memory usage statistics
    mutable size_t peak_state_usage_ = 0;
    mutable size_t peak_control_usage_ = 0;
    mutable size_t peak_action_usage_ = 0;
    mutable size_t peak_primitive_usage_ = 0;
    mutable size_t peak_footprint_usage_ = 0;
    
public:
    NAMOMemoryManager() = default;
    
    // State management
    State* get_state() {
        State* state = state_pool_.acquire();
        peak_state_usage_ = std::max(peak_state_usage_, state_pool_.used_count());
        return state;
    }
    
    void return_state(State* state) {
        state_pool_.release(state);
    }
    
    // Control management
    Control* get_control() {
        Control* control = control_pool_.acquire();
        peak_control_usage_ = std::max(peak_control_usage_, control_pool_.used_count());
        return control;
    }
    
    void return_control(Control* control) {
        control_pool_.release(control);
    }
    
    // Action management
    ActionStepMPC* get_action() {
        ActionStepMPC* action = action_pool_.acquire();
        peak_action_usage_ = std::max(peak_action_usage_, action_pool_.used_count());
        return action;
    }
    
    void return_action(ActionStepMPC* action) {
        action_pool_.release(action);
    }
    
    // Motion primitive management
    MotionPrimitive* get_primitive() {
        MotionPrimitive* primitive = primitive_pool_.acquire();
        peak_primitive_usage_ = std::max(peak_primitive_usage_, primitive_pool_.used_count());
        return primitive;
    }
    
    void return_primitive(MotionPrimitive* primitive) {
        primitive_pool_.release(primitive);
    }
    
    // Grid footprint management
    GridFootprint* get_footprint() {
        GridFootprint* footprint = footprint_pool_.acquire();
        peak_footprint_usage_ = std::max(peak_footprint_usage_, footprint_pool_.used_count());
        return footprint;
    }
    
    void return_footprint(GridFootprint* footprint) {
        footprint_pool_.release(footprint);
    }
    
    /**
     * @brief Reset all memory pools
     */
    void reset_all() {
        state_pool_.reset_all();
        control_pool_.reset_all();
        action_pool_.reset_all();
        primitive_pool_.reset_all();
        footprint_pool_.reset_all();
    }
    
    /**
     * @brief Get memory usage statistics
     */
    void print_statistics() const {
        std::cout << "=== Memory Pool Statistics ===" << std::endl;
        std::cout << "States: " << state_pool_.used_count() << "/" << state_pool_.size() 
                  << " (peak: " << peak_state_usage_ << ")" << std::endl;
        std::cout << "Controls: " << control_pool_.used_count() << "/" << control_pool_.size() 
                  << " (peak: " << peak_control_usage_ << ")" << std::endl;
        std::cout << "Actions: " << action_pool_.used_count() << "/" << action_pool_.size() 
                  << " (peak: " << peak_action_usage_ << ")" << std::endl;
        std::cout << "Primitives: " << primitive_pool_.used_count() << "/" << primitive_pool_.size() 
                  << " (peak: " << peak_primitive_usage_ << ")" << std::endl;
        std::cout << "Footprints: " << footprint_pool_.used_count() << "/" << footprint_pool_.size() 
                  << " (peak: " << peak_footprint_usage_ << ")" << std::endl;
    }
    
    /**
     * @brief Check if any pools are exhausted
     */
    bool check_pool_health() const {
        bool healthy = true;
        
        if (state_pool_.is_exhausted()) {
            std::cerr << "WARNING: State pool exhausted!" << std::endl;
            healthy = false;
        }
        
        if (control_pool_.is_exhausted()) {
            std::cerr << "WARNING: Control pool exhausted!" << std::endl;
            healthy = false;
        }
        
        if (action_pool_.is_exhausted()) {
            std::cerr << "WARNING: Action pool exhausted!" << std::endl;
            healthy = false;
        }
        
        if (primitive_pool_.is_exhausted()) {
            std::cerr << "WARNING: Primitive pool exhausted!" << std::endl;
            healthy = false;
        }
        
        if (footprint_pool_.is_exhausted()) {
            std::cerr << "WARNING: Footprint pool exhausted!" << std::endl;
            healthy = false;
        }
        
        return healthy;
    }
    
    /**
     * @brief Get available capacity for each pool
     */
    struct PoolCapacity {
        size_t states_available;
        size_t controls_available;
        size_t actions_available;
        size_t primitives_available;
        size_t footprints_available;
    };
    
    PoolCapacity get_available_capacity() const {
        return {
            state_pool_.available_count(),
            control_pool_.available_count(),
            action_pool_.available_count(),
            primitive_pool_.available_count(),
            footprint_pool_.available_count()
        };
    }
};

/**
 * @brief RAII wrapper for automatic memory management
 * 
 * Automatically returns objects to pools when they go out of scope
 */
template<typename T>
class PooledPtr {
private:
    T* ptr_;
    std::function<void(T*)> deleter_;
    
public:
    PooledPtr(T* ptr, std::function<void(T*)> deleter) 
        : ptr_(ptr), deleter_(deleter) {}
    
    ~PooledPtr() {
        if (ptr_) {
            deleter_(ptr_);
        }
    }
    
    // Disable copy, enable move
    PooledPtr(const PooledPtr&) = delete;
    PooledPtr& operator=(const PooledPtr&) = delete;
    
    PooledPtr(PooledPtr&& other) noexcept 
        : ptr_(other.ptr_), deleter_(std::move(other.deleter_)) {
        other.ptr_ = nullptr;
    }
    
    PooledPtr& operator=(PooledPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) deleter_(ptr_);
            ptr_ = other.ptr_;
            deleter_ = std::move(other.deleter_);
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    
    T* release() {
        T* result = ptr_;
        ptr_ = nullptr;
        return result;
    }
};

// Convenience functions for creating pooled pointers
inline PooledPtr<State> make_pooled_state(NAMOMemoryManager& manager) {
    State* state = manager.get_state();
    return PooledPtr<State>(state, [&manager](State* s) { manager.return_state(s); });
}

inline PooledPtr<Control> make_pooled_control(NAMOMemoryManager& manager) {
    Control* control = manager.get_control();
    return PooledPtr<Control>(control, [&manager](Control* c) { manager.return_control(c); });
}

inline PooledPtr<ActionStepMPC> make_pooled_action(NAMOMemoryManager& manager) {
    ActionStepMPC* action = manager.get_action();
    return PooledPtr<ActionStepMPC>(action, [&manager](ActionStepMPC* a) { manager.return_action(a); });
}

} // namespace namo