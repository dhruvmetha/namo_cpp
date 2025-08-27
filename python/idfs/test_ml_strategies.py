"""Tests for ML-based selection strategies.

This module provides comprehensive tests for the ML-based object selection
and goal generation strategies, including fallback behavior and error handling.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import Counter

# Ensure namo_rl is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import namo_rl

# Import strategies to test
from idfs.ml_strategies import MLObjectSelectionStrategy, MLGoalSelectionStrategy
from idfs.object_selection_strategy import NearestFirstStrategy
from idfs.goal_selection_strategy import RandomGoalStrategy, Goal


class TestMLObjectSelectionStrategy:
    """Test cases for MLObjectSelectionStrategy."""
    
    @pytest.fixture
    def mock_env(self):
        """Create mock environment for testing."""
        env = Mock(spec=namo_rl.RLEnvironment)
        
        # Mock state management
        mock_state = Mock(spec=namo_rl.RLState)
        env.get_full_state.return_value = mock_state
        env.set_full_state.return_value = None
        
        # Mock observations
        env.get_observation.return_value = {
            'robot_pose': [0.0, 0.0, 0.1],
            'box1_pose': [1.2, 1.8, 0.05],
            'box2_pose': [2.1, 0.5, 0.05],
            'robot_goal': [1.5, 2.0]
        }
        
        # Mock robot goal
        env._robot_goal = [1.5, 2.0]
        
        return env
    
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        strategy = MLObjectSelectionStrategy("fake/model/path")
        
        assert strategy.object_model_path == "fake/model/path"
        assert strategy.fallback_strategy is None
        assert strategy.samples == 32
        assert strategy.device == "cuda"
        assert strategy.verbose is False
        assert strategy._object_model is None
        assert strategy._load_attempted is False
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = MLObjectSelectionStrategy(
            object_model_path="custom/path",
            samples=16,
            device="cpu",
            verbose=True,
            min_valid_samples=10
        )
        
        assert strategy.object_model_path == "custom/path"
        assert strategy.samples == 16
        assert strategy.device == "cpu"
        assert strategy.verbose is True
        assert strategy.min_valid_samples == 10
    
    def test_strategy_name(self):
        """Test strategy name generation."""
        strategy = MLObjectSelectionStrategy("fake/path")
        assert strategy.strategy_name == "ML Object Selection"
    
    @patch('idfs.ml_strategies.sys.path')
    def test_load_model_success(self, mock_path):
        """Test successful model loading."""
        strategy = MLObjectSelectionStrategy("fake/path", verbose=True)
        
        # Mock the import and model creation
        with patch('importlib.import_module') as mock_import:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_class.return_value = mock_model_instance
            
            mock_module = Mock()
            mock_module.ObjectInferenceModel = mock_model_class
            mock_import.return_value = mock_module
            
            # Patch the actual import that happens in _load_model
            with patch('idfs.ml_strategies.ObjectInferenceModel', mock_model_class):
                result = strategy._load_model()
        
        assert result is True
        assert strategy._object_model == mock_model_instance
        assert strategy._load_attempted is True
        mock_model_class.assert_called_once_with(
            model_path="fake/path",
            device="cuda"
        )
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        strategy = MLObjectSelectionStrategy("fake/path", verbose=True)
        
        # Mock import failure
        with patch('idfs.ml_strategies.ObjectInferenceModel', side_effect=ImportError("No module")):
            result = strategy._load_model()
        
        assert result is False
        assert strategy._object_model is None
        assert strategy._load_attempted is True
    
    def test_create_json_message_success(self, mock_env):
        """Test successful JSON message creation."""
        strategy = MLObjectSelectionStrategy("fake/path")
        reachable_objects = ["box1", "box2"]
        mock_state = Mock()
        
        json_msg = strategy._create_json_message(reachable_objects, mock_state, mock_env)
        
        assert json_msg is not None
        assert json_msg["robot_goal"] == [1.5, 2.0]
        assert json_msg["robot"]["position"] == [0.0, 0.0, 0.1]
        assert len(json_msg["objects"]) == 2
        assert "box1" in json_msg["objects"]
        assert "box2" in json_msg["objects"]
        assert json_msg["reachable_objects"] == reachable_objects
    
    def test_create_json_message_missing_robot_pose(self, mock_env):
        """Test JSON message creation with missing robot pose."""
        strategy = MLObjectSelectionStrategy("fake/path")
        reachable_objects = ["box1"]
        mock_state = Mock()
        
        # Remove robot_pose from observations
        mock_env.get_observation.return_value = {
            'box1_pose': [1.2, 1.8, 0.05],
        }
        
        json_msg = strategy._create_json_message(reachable_objects, mock_state, mock_env)
        
        assert json_msg is None
    
    def test_create_json_message_missing_robot_goal(self, mock_env):
        """Test JSON message creation with missing robot goal."""
        strategy = MLObjectSelectionStrategy("fake/path")
        reachable_objects = ["box1"]
        mock_state = Mock()
        
        # Remove robot goal
        mock_env.get_observation.return_value = {
            'robot_pose': [0.0, 0.0, 0.1],
            'box1_pose': [1.2, 1.8, 0.05],
        }
        del mock_env._robot_goal
        
        json_msg = strategy._create_json_message(reachable_objects, mock_state, mock_env)
        
        assert json_msg is None
    
    def test_select_objects_empty_list(self, mock_env):
        """Test behavior with empty object list."""
        strategy = MLObjectSelectionStrategy("fake/path")
        mock_state = Mock()
        
        result = strategy.select_objects([], mock_state, mock_env)
        
        assert result == []
    
    def test_select_objects_ml_success(self, mock_env):
        """Test successful ML object selection."""
        strategy = MLObjectSelectionStrategy("fake/path", verbose=True)
        reachable_objects = ["box1", "box2", "box3"]
        mock_state = Mock()
        
        # Mock successful model loading and inference
        with patch.object(strategy, '_load_model', return_value=True):
            with patch.object(strategy, '_create_json_message', return_value={"fake": "message"}):
                # Mock model inference
                mock_model = Mock()
                strategy._object_model = mock_model
                
                mock_model.infer.return_value = {
                    'total_valid_samples': 20,
                    'object_votes': Counter({'box2': 15, 'box1': 10, 'box3': 5})
                }
                
                result = strategy.select_objects(reachable_objects, mock_state, mock_env)
        
        # Should be sorted by vote count: box2 (15), box1 (10), box3 (5)
        assert result == ["box2", "box1", "box3"]
    
    def test_select_objects_ml_insufficient_samples(self, mock_env):
        """Test ML failure due to insufficient samples."""
        strategy = MLObjectSelectionStrategy(
            "fake/path", 
            verbose=True
        )
        reachable_objects = ["box1", "box2"]
        mock_state = Mock()
        
        # Mock model loading success but insufficient samples
        with patch.object(strategy, '_load_model', return_value=True):
            with patch.object(strategy, '_create_json_message', return_value={"fake": "message"}):
                mock_model = Mock()
                strategy._object_model = mock_model
                
                mock_model.infer.return_value = {
                    'total_valid_samples': 2,  # Below min_valid_samples (5)
                    'object_votes': Counter({'box1': 1, 'box2': 1})
                }
                
                result = strategy.select_objects(reachable_objects, mock_state, mock_env)
        
        # Should return original order when ML fails
        assert result == ["box1", "box2"]
    
    def test_select_objects_model_load_failure(self, mock_env):
        """Test model loading failure."""
        strategy = MLObjectSelectionStrategy(
            "fake/path", 
            verbose=True
        )
        reachable_objects = ["box1", "box2"]
        mock_state = Mock()
        
        # Mock model loading failure
        with patch.object(strategy, '_load_model', return_value=False):
            result = strategy.select_objects(reachable_objects, mock_state, mock_env)
        
        # Should return original order when ML fails
        assert result == ["box1", "box2"]
    


class TestMLGoalSelectionStrategy:
    """Test cases for MLGoalSelectionStrategy."""
    
    @pytest.fixture
    def mock_env(self):
        """Create mock environment for testing."""
        env = Mock(spec=namo_rl.RLEnvironment)
        
        # Mock state management
        mock_state = Mock(spec=namo_rl.RLState)
        env.get_full_state.return_value = mock_state
        env.set_full_state.return_value = None
        
        # Mock observations
        env.get_observation.return_value = {
            'robot_pose': [0.0, 0.0, 0.1],
            'box1_pose': [1.2, 1.8, 0.05],
            'box2_pose': [2.1, 0.5, 0.05],
        }
        
        # Mock robot goal
        env._robot_goal = [1.5, 2.0]
        
        return env
    
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        strategy = MLGoalSelectionStrategy("goal/path")
        
        assert strategy.goal_model_path == "goal/path"
        assert strategy.samples == 32
        assert strategy.device == "cuda"
        assert strategy.verbose is False
        assert strategy._goal_model is None
        assert strategy._load_attempted is False
    
    def test_strategy_name(self):
        """Test strategy name generation."""
        strategy = MLGoalSelectionStrategy("goal/path")
        assert strategy.strategy_name == "ML Goal Generation"
    
    def test_load_model_success(self):
        """Test successful model loading."""
        strategy = MLGoalSelectionStrategy("goal/path", verbose=True)
        
        # Mock model class
        mock_goal_model = Mock()
        mock_goal_instance = Mock()
        
        mock_goal_model.return_value = mock_goal_instance
        
        with patch('idfs.ml_strategies.GoalInferenceModel', mock_goal_model):
            result = strategy._load_model()
        
        assert result is True
        assert strategy._goal_model == mock_goal_instance
        assert strategy._load_attempted is True
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        strategy = MLGoalSelectionStrategy("goal/path", verbose=True)
        
        # Mock import failure
        with patch('idfs.ml_strategies.GoalInferenceModel', side_effect=ImportError("No module")):
            result = strategy._load_model()
        
        assert result is False
        assert strategy._goal_model is None
        assert strategy._load_attempted is True
    
    def test_generate_goals_empty_object(self, mock_env):
        """Test behavior when object doesn't exist."""
        strategy = MLGoalSelectionStrategy("goal/path", "object/path")
        mock_state = Mock()
        
        # Object not in observations
        result = strategy.generate_goals("nonexistent_object", mock_state, mock_env, 5)
        
        assert result == []
    
    def test_generate_goals_ml_success(self, mock_env):
        """Test successful ML goal generation."""
        strategy = MLGoalSelectionStrategy("goal/path", verbose=True)
        mock_state = Mock()
        
        # Mock successful model loading
        with patch.object(strategy, '_load_model', return_value=True):
            with patch.object(strategy, '_create_json_message_for_goals', 
                            return_value={"fake": "message", "xml_path": "test.xml", "robot_goal": [1.0, 2.0]}):
                
                # Mock goal model
                mock_goal_model = Mock()
                strategy._goal_model = mock_goal_model
                
                # Mock goal model inference with new independent API
                mock_goal_model.infer.return_value = [
                    {'x': 1.5, 'y': 2.0, 'theta': 0.5},
                    {'x': 1.8, 'y': 2.2, 'theta': 1.0},
                    {'x': 1.2, 'y': 1.8, 'theta': 0.0}
                ]
                
                result = strategy.generate_goals("box1", mock_state, mock_env, 5)
        
        assert len(result) == 3
        assert all(isinstance(goal, Goal) for goal in result)
        assert result[0].x == 1.5
        assert result[0].y == 2.0
        assert result[0].theta == 0.5
    
    def test_generate_goals_insufficient_goals(self, mock_env):
        """Test ML failure due to insufficient goals."""
        strategy = MLGoalSelectionStrategy(
            "goal/path",
            min_goals_threshold=3,
            verbose=True
        )
        mock_state = Mock()
        
        # Mock model loading success but insufficient goals
        with patch.object(strategy, '_load_model', return_value=True):
            with patch.object(strategy, '_create_json_message_for_goals', 
                            return_value={"fake": "message", "xml_path": "test.xml", "robot_goal": [1.0, 2.0]}):
                
                mock_goal_model = Mock()
                strategy._goal_model = mock_goal_model
                
                # Return too few goals
                mock_goal_model.infer.return_value = [
                    {'x': 1.5, 'y': 2.0, 'theta': 0.5}
                ]  # Only 1 goal, but threshold is 3
                
                result = strategy.generate_goals("box1", mock_state, mock_env, 5)
        
        # Should return empty list when ML fails
        assert result == []
    
    def test_generate_goals_model_load_failure(self, mock_env):
        """Test behavior when ML model loading fails."""
        strategy = MLGoalSelectionStrategy("goal/path", verbose=True)
        mock_state = Mock()
        
        # Mock model loading failure
        with patch.object(strategy, '_load_model', return_value=False):
            result = strategy.generate_goals("box1", mock_state, mock_env, 5)
        
        # Should return empty list
        assert result == []


class TestIntegrationScenarios:
    """Integration tests for ML strategies with realistic scenarios."""
    
    def test_object_selection_with_partial_votes(self):
        """Test object selection when ML model doesn't vote for all objects."""
        strategy = MLObjectSelectionStrategy("fake/path", verbose=True)
        mock_env = Mock()
        mock_state = Mock()
        reachable_objects = ["box1", "box2", "box3", "box4"]
        
        # Mock successful ML inference with partial votes
        with patch.object(strategy, '_load_model', return_value=True):
            with patch.object(strategy, '_create_json_message', return_value={"fake": "message"}):
                mock_model = Mock()
                strategy._object_model = mock_model
                
                # Model only votes for 2 out of 4 objects
                mock_model.infer.return_value = {
                    'total_valid_samples': 20,
                    'object_votes': Counter({'box3': 12, 'box1': 8})  # Missing box2, box4
                }
                
                result = strategy.select_objects(reachable_objects, mock_state, mock_env)
        
        # Voted objects should come first, then unvoted ones in original order
        assert result[:2] == ["box3", "box1"]  # Sorted by votes
        assert "box2" in result[2:]  # Unvoted objects at the end
        assert "box4" in result[2:]
        assert len(result) == 4  # All objects included
    
    def test_goal_generation_max_goals_limit(self, mock_env=None):
        """Test that goal generation respects max_goals limit."""
        if mock_env is None:
            mock_env = Mock()
            mock_env.get_full_state.return_value = Mock()
            mock_env.get_observation.return_value = {
                'robot_pose': [0.0, 0.0, 0.1],
                'box1_pose': [1.2, 1.8, 0.05],
            }
            mock_env._robot_goal = [1.5, 2.0]
        
        strategy = MLGoalSelectionStrategy("goal/path", "object/path")
        mock_state = Mock()
        
        with patch.object(strategy, '_load_models', return_value=True):
            with patch.object(strategy, '_create_json_message_for_goals', 
                            return_value={"fake": "message"}):
                
                mock_object_model = Mock()
                mock_goal_model = Mock()
                strategy._object_model = mock_object_model
                strategy._goal_model = mock_goal_model
                
                mock_object_model.infer.return_value = {
                    'inp_data': {'fake': 'data'},
                    'image_converter': Mock()
                }
                
                # Return many goals
                mock_goal_model.infer.return_value = [
                    {'x': i * 0.1, 'y': i * 0.2, 'theta': i * 0.3} 
                    for i in range(10)
                ]  # 10 goals
                
                result = strategy.generate_goals("box1", mock_state, mock_env, max_goals=3)
        
        # Should respect max_goals limit
        assert len(result) == 3


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])