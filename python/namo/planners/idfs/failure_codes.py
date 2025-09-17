#!/usr/bin/env python3
"""Failure Classification System for NAMO Data Collection

This module provides standardized failure codes and classification utilities
for tracking why episodes fail during data collection. This enables better
debugging and analysis of failure patterns.
"""

from enum import IntEnum
from typing import Dict, Any, Optional
import traceback
import re


class FailureCode(IntEnum):
    """Standardized failure codes for episode failures."""
    
    # Success (not a failure, but included for completeness)
    SUCCESS = 0
    
    # Planning-related failures
    TIMEOUT = 1                    # Search exceeded time limit
    MAX_TERMINAL_CHECKS = 2        # Exceeded maximum terminal checks
    MAX_DEPTH_REACHED = 3          # Reached maximum search depth without solution
    NO_REACHABLE_OBJECTS = 4       # No objects are reachable from current state
    
    # Environment/execution failures
    ROBOT_COLLISION = 5            # Robot collided with object or wall
    OBJECT_COLLISION = 6           # Object-to-object collision during push
    INVALID_ACTION = 7             # Action parameters are invalid
    ACTION_EXECUTION_FAILED = 8    # Action failed to execute properly
    
    # Environment setup failures
    ENVIRONMENT_LOAD_FAILED = 9    # Failed to load XML environment
    ENVIRONMENT_RESET_FAILED = 10  # Failed to reset environment
    GOAL_SET_FAILED = 11          # Failed to set robot goal
    
    # ML model failures
    ML_MODEL_LOAD_FAILED = 12     # Failed to load ML model
    ML_INFERENCE_FAILED = 13      # ML inference crashed
    
    # Data/state failures  
    STATE_EXTRACTION_FAILED = 14  # Failed to get state observations
    INVALID_STATE = 15            # State is corrupted or invalid
    
    # System/resource failures
    MEMORY_ERROR = 16             # Out of memory
    SYSTEM_ERROR = 17             # System-level error (disk, network, etc.)
    
    # Unknown/unclassified failures
    UNKNOWN_ERROR = 99            # Catch-all for unclassified errors


class FailureClassifier:
    """Utility class for classifying failures based on exception messages."""
    
    # Regex patterns for common error messages
    FAILURE_PATTERNS = {
        FailureCode.TIMEOUT: [
            r"timeout",
            r"time.*out",
            r"exceeded.*time",
            r"search.*timeout",
            r"planning.*timeout",
            r"execution.*timeout",
            r"time.*limit.*exceeded"
        ],
        FailureCode.MAX_TERMINAL_CHECKS: [
            r"terminal.*check.*limit",
            r"exceeded.*terminal",
            r"max.*terminal.*checks"
        ],
        FailureCode.MAX_DEPTH_REACHED: [
            r"max.*depth.*reached",
            r"depth.*limit.*exceeded",
            r"maximum.*depth"
        ],
        FailureCode.NO_REACHABLE_OBJECTS: [
            r"no.*reachable.*object",
            r"no.*reachable.*edge",
            r"no.*reachable.*edges",
            r"reachable.*objects.*empty",
            r"no.*objects.*available",
            r"reachable.*edge.*idx.*found",
            r"no.*reachable.*edge.*idx"
        ],
        FailureCode.ROBOT_COLLISION: [
            r"robot.*collision",
            r"robot.*contact",
            r"robot.*hit"
        ],
        FailureCode.OBJECT_COLLISION: [
            r"object.*collision",
            r"object.*contact",
            r"collision.*detected"
        ],
        FailureCode.INVALID_ACTION: [
            r"invalid.*action",
            r"action.*invalid",
            r"bad.*action.*parameter"
        ],
        FailureCode.ACTION_EXECUTION_FAILED: [
            r"action.*execution.*failed",
            r"execute.*action.*failed",
            r"push.*failed",
            r"manipulation.*failed",
            r"mpc.*execution.*failed",
            r"skill.*execution.*failed",
            r"failed.*execute.*skill"
        ],
        FailureCode.ENVIRONMENT_LOAD_FAILED: [
            r"environment.*load.*failed",
            r"xml.*load.*error",
            r"scene.*load.*failed",
            r"mujoco.*load.*error"
        ],
        FailureCode.ENVIRONMENT_RESET_FAILED: [
            r"environment.*reset.*failed",
            r"env.*reset.*error",
            r"reset.*failed"
        ],
        FailureCode.GOAL_SET_FAILED: [
            r"goal.*set.*failed",
            r"set.*goal.*error",
            r"robot.*goal.*failed"
        ],
        FailureCode.ML_MODEL_LOAD_FAILED: [
            r"ml.*model.*load.*failed",
            r"model.*load.*error",
            r"failed.*load.*ml.*model",
            r"failed.*load.*pytorch.*model",
            r"failed.*load.*torch.*model",
            r"inference.*model.*failed",
            r"pytorch.*load.*error",
            r"torch.*load.*error"
        ],
        FailureCode.ML_INFERENCE_FAILED: [
            r"ml.*inference.*failed",
            r"model.*prediction.*failed",
            r"forward.*pass.*failed",
            r"inference.*error"
        ],
        FailureCode.STATE_EXTRACTION_FAILED: [
            r"state.*extract.*failed",
            r"observation.*failed",
            r"get.*state.*error",
            r"state.*observation.*error"
        ],
        FailureCode.INVALID_STATE: [
            r"invalid.*state",
            r"state.*corrupted",
            r"state.*invalid",
            r"bad.*state"
        ],
        FailureCode.MEMORY_ERROR: [
            r"memory.*error",
            r"out.*of.*memory",
            r"allocation.*failed",
            r"memoryerror"
        ],
        FailureCode.SYSTEM_ERROR: [
            r"system.*error",
            r"os.*error",
            r"disk.*error",
            r"io.*error",
            r"network.*error"
        ]
    }
    
    @classmethod
    def classify_failure(cls, error_message: str, exception: Optional[Exception] = None) -> FailureCode:
        """
        Classify failure based on error message and optional exception.
        
        Args:
            error_message: String description of the error
            exception: Optional exception object that caused the failure
            
        Returns:
            FailureCode enum value representing the classified failure type
        """
        if not error_message:
            error_message = ""
            
        # Convert to lowercase for pattern matching
        error_lower = error_message.lower()
        
        # Check exception type first if provided
        if exception is not None:
            if isinstance(exception, MemoryError):
                return FailureCode.MEMORY_ERROR
            elif isinstance(exception, TimeoutError):
                return FailureCode.TIMEOUT
            elif isinstance(exception, (OSError, IOError)):
                return FailureCode.SYSTEM_ERROR
        
        # Check patterns in order of specificity (most specific first)
        for failure_code, patterns in cls.FAILURE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return failure_code
        
        # If no pattern matches, return unknown error
        return FailureCode.UNKNOWN_ERROR
    
    @classmethod
    def get_failure_description(cls, failure_code: FailureCode) -> str:
        """Get human-readable description of failure code."""
        descriptions = {
            FailureCode.SUCCESS: "Episode completed successfully",
            FailureCode.TIMEOUT: "Search exceeded time limit",
            FailureCode.MAX_TERMINAL_CHECKS: "Exceeded maximum terminal state checks",
            FailureCode.MAX_DEPTH_REACHED: "Reached maximum search depth without solution",
            FailureCode.NO_REACHABLE_OBJECTS: "No reachable objects available for manipulation",
            FailureCode.ROBOT_COLLISION: "Robot collided with object or environment",
            FailureCode.OBJECT_COLLISION: "Object collision during manipulation",
            FailureCode.INVALID_ACTION: "Action parameters are invalid",
            FailureCode.ACTION_EXECUTION_FAILED: "Failed to execute manipulation action",
            FailureCode.ENVIRONMENT_LOAD_FAILED: "Failed to load environment from XML",
            FailureCode.ENVIRONMENT_RESET_FAILED: "Failed to reset environment state",
            FailureCode.GOAL_SET_FAILED: "Failed to set robot goal position",
            FailureCode.ML_MODEL_LOAD_FAILED: "Failed to load ML inference model",
            FailureCode.ML_INFERENCE_FAILED: "ML model inference failed",
            FailureCode.STATE_EXTRACTION_FAILED: "Failed to extract state observations",
            FailureCode.INVALID_STATE: "Environment state is invalid or corrupted",
            FailureCode.MEMORY_ERROR: "Insufficient memory for operation",
            FailureCode.SYSTEM_ERROR: "System-level error (I/O, network, etc.)",
            FailureCode.UNKNOWN_ERROR: "Unclassified error"
        }
        return descriptions.get(failure_code, f"Unknown failure code: {failure_code}")


def classify_exception(exception: Exception) -> FailureCode:
    """
    Convenience function to classify an exception object.
    
    Args:
        exception: Exception object to classify
        
    Returns:
        FailureCode enum value
    """
    error_message = str(exception)
    return FailureClassifier.classify_failure(error_message, exception)


def create_failure_info(error_message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
    """
    Create comprehensive failure information dictionary.
    
    Args:
        error_message: String description of the error
        exception: Optional exception object
        
    Returns:
        Dictionary containing failure code, description, and traceback info
    """
    failure_code = FailureClassifier.classify_failure(error_message, exception)
    
    failure_info = {
        'failure_code': int(failure_code),
        'failure_description': FailureClassifier.get_failure_description(failure_code),
        'error_message': error_message,
    }
    
    # Add traceback if exception is provided
    if exception is not None:
        failure_info['exception_type'] = type(exception).__name__
        failure_info['traceback'] = traceback.format_exception(type(exception), exception, exception.__traceback__)
    
    return failure_info


def get_failure_statistics(episode_results: list) -> Dict[str, Any]:
    """
    Analyze failure patterns across a collection of episode results.
    
    Args:
        episode_results: List of episode result dictionaries
        
    Returns:
        Dictionary with failure statistics and patterns
    """
    failure_counts = {}
    total_episodes = len(episode_results)
    successful_episodes = 0
    
    for episode in episode_results:
        if episode.get('success', False) and episode.get('solution_found', False):
            successful_episodes += 1
            continue
            
        # Get failure code from episode result
        failure_code = episode.get('failure_code', FailureCode.UNKNOWN_ERROR)
        
        # Handle None values explicitly
        if failure_code is None:
            failure_code = FailureCode.UNKNOWN_ERROR
        elif isinstance(failure_code, int):
            failure_code = FailureCode(failure_code)
            
        failure_counts[failure_code] = failure_counts.get(failure_code, 0) + 1
    
    # Convert to readable format
    failure_breakdown = {}
    for failure_code, count in failure_counts.items():
        description = FailureClassifier.get_failure_description(failure_code)
        percentage = (count / total_episodes) * 100 if total_episodes > 0 else 0
        failure_breakdown[description] = {
            'count': count,
            'percentage': percentage,
            'failure_code': int(failure_code) if failure_code is not None else FailureCode.UNKNOWN_ERROR.value
        }
    
    success_rate = (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0
    
    return {
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes,
        'failed_episodes': total_episodes - successful_episodes,
        'success_rate': success_rate,
        'failure_breakdown': failure_breakdown,
        'most_common_failure': max(failure_counts.keys(), key=failure_counts.get) if failure_counts else None
    }


if __name__ == "__main__":
    # Example usage
    print("NAMO Failure Classification System")
    print("=" * 50)
    
    # Test classification of different error types
    test_errors = [
        "Search timeout exceeded after 300 seconds",
        "No reachable objects available for manipulation", 
        "Robot collision detected during push action",
        "Failed to load ML model from path",
        "Maximum depth of 5 reached without solution",
        "Some random unknown error occurred"
    ]
    
    for error in test_errors:
        code = FailureClassifier.classify_failure(error)
        desc = FailureClassifier.get_failure_description(code)
        print(f"'{error}' -> {code.name} ({code.value}): {desc}")