"""ML-based selection strategies for IDFS planners.

This module provides ML-based strategies for object selection and goal generation
using trained diffusion models from the learning package.
"""

import sys
import os
import random
from typing import List, Optional, Dict, Any
from collections import Counter
import namo_rl
from .object_selection_strategy import ObjectSelectionStrategy
from .goal_selection_strategy import GoalSelectionStrategy, Goal, RandomGoalStrategy

# Add learning package to path for imports
learning_path = "/common/home/dm1487/robotics_research/ktamp/learning"
if learning_path not in sys.path:
    sys.path.append(learning_path)


class MLObjectSelectionStrategy(ObjectSelectionStrategy):
    """ML-based object selection using ObjectInferenceModel.
    
    Uses trained diffusion models to select objects based on learned preferences
    from visual scene analysis and robot goal context.
    """
    
    def __init__(self,
                 object_model_path: str,
                 samples: int = 32,
                 device: str = "cuda",
                 xml_path: Optional[str] = None,
                 min_valid_samples: int = 1,
                 verbose: bool = False,
                 preloaded_model: Optional[Any] = None):
        """Initialize ML object selection strategy.

        Args:
            object_model_path: Path to trained object inference model
            samples: Number of diffusion samples for inference
            device: PyTorch device ("cuda" or "cpu")
            xml_path: XML file path (absolute or relative). If None, uses env.get_xml_path()
            min_valid_samples: Minimum valid samples before considering result reliable
            verbose: Enable verbose logging
            preloaded_model: Optional pre-loaded model to avoid repeated loading
        """
        self.object_model_path = object_model_path
        self.samples = samples
        self.device = device
        self.xml_path = xml_path
        self.min_valid_samples = min_valid_samples
        self.verbose = False # verbose
        
        # Use preloaded model if available, otherwise lazy load
        if preloaded_model is not None:
            self._object_model = preloaded_model
            self._load_attempted = True
            if self.verbose:
                print("Using preloaded ObjectInferenceModel")
        else:
            self._object_model = None
            self._load_attempted = False
    
    def _load_model(self):
        """Lazy load the object inference model."""
        if self._load_attempted:
            return self._object_model is not None
        
        self._load_attempted = True
        
        print(f"Loading ObjectInferenceModel from {self.object_model_path}...")
        try:
            from ktamp_learning.object_inference_model import ObjectInferenceModel
            
            if self.verbose:
                print(f"Loading ObjectInferenceModel from {self.object_model_path}")
            
            self._object_model = ObjectInferenceModel(
                model_path=self.object_model_path,
                device=self.device
            )
            
            print("ObjectInferenceModel loaded successfully")
            if self.verbose:
                print("ObjectInferenceModel loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to load ObjectInferenceModel: {e}")
            if self.verbose:
                print(f"Failed to load ObjectInferenceModel: {e}")
            self._object_model = None
            return False
    
    def _create_json_message(self, 
                           reachable_objects: List[str],
                           state: namo_rl.RLState, 
                           env: namo_rl.RLEnvironment) -> Optional[Dict[str, Any]]:
        """Create JSON message format expected by ObjectInferenceModel."""
        # Save original state to restore later
        original_state = env.get_full_state()
        
        try:
            # Set state to get observations
            env.set_full_state(state)
            obs = env.get_observation()
            
            # Get robot position and goal
            robot_pose = obs.get('robot_pose')
            if robot_pose is None or len(robot_pose) < 3:
                return None
            
            # Get robot goal - try multiple methods
            robot_goal = None
            if hasattr(env, '_robot_goal'):
                robot_goal = getattr(env, '_robot_goal')
            elif 'robot_goal' in obs:
                robot_goal = obs['robot_goal']
            elif hasattr(env, 'get_robot_goal'):
                try:
                    robot_goal = env.get_robot_goal()
                except:
                    pass
            
            if robot_goal is None or len(robot_goal) < 2:
                return None
            
            # Build objects dictionary from observations
            objects_dict = {}
            for obj_id in reachable_objects:
                pose_key = f"{obj_id}_pose"
                if pose_key in obs:
                    pose = obs[pose_key]
                    if len(pose) >= 3:
                        # Convert theta angle (pose[2]) to quaternion
                        from scipy.spatial.transform import Rotation as R
                        import numpy as np
                        quat = R.from_euler('xyz', [0, 0, pose[2]], degrees=False).as_quat(scalar_first=True)
                        objects_dict[obj_id] = {
                            "position": [float(pose[0]), float(pose[1]), float(pose[2])],
                            "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                        }
            
            # Add static objects from get_object_info()
            try:
                static_info = env.get_object_info()
                for obj_name, obj_data in static_info.items():
                    if obj_name not in objects_dict:  # Don't override movable objects
                        # Only add objects that have position data and are actually static walls
                        pos_x = obj_data.get('pos_x', None)
                        pos_y = obj_data.get('pos_y', None)
                        
                        # Skip objects without position data (like movable objects that only have size info)
                        # Also skip the robot since it's not a static obstacle
                        if pos_x is None or pos_y is None or obj_name == 'robot':
                            continue
                            
                        angle_deg = obj_data.get('angle_deg', 0.0)
                        
                        # Convert angle from degrees to quaternion
                        quat = R.from_euler('xyz', [0, 0, angle_deg], degrees=True).as_quat(scalar_first=True)
                        objects_dict[obj_name] = {
                            "position": [float(pos_x), float(pos_y), float(angle_deg * np.pi / 180.0)],  # Keep theta in position for compatibility
                            "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                        }
            except Exception as e:
                # If get_object_info fails, continue without static objects
                pass
            
            if not objects_dict:
                return None
            
            # Get XML path - use provided path or get from environment
            xml_path = self.xml_path
            if xml_path is None:
                # Try to get from environment
                if hasattr(env, 'get_xml_path'):
                    xml_path = env.get_xml_path()
                else:
                    print(f"  ⚠️ JSON creation failed: No XML path provided and env.get_xml_path() not available")
                    return None
            
            # Create message in expected format
            json_message = {
                "xml_path": xml_path,
                "robot_goal": [float(robot_goal[0]), float(robot_goal[1])],
                "reachable_objects": reachable_objects,
                "robot": {
                    "position": [float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])]
                },
                "objects": objects_dict
            }
            
            return json_message
            
        finally:
            # Always restore original state
            env.set_full_state(original_state)
    
    def select_objects(self, 
                      reachable_objects: List[str], 
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment) -> List[str]:
        """Select objects using ML model."""
        # Quick exit for empty list
        if not reachable_objects:
            return []
        
        # Try ML inference
        ml_result = self._select_objects_ml(reachable_objects, state, env)
        if ml_result:
            return ml_result
        
        # If ML fails, return original order
        if self.verbose:
            print("ML object selection failed, using original order")
        return reachable_objects.copy()
    
    def _select_objects_ml(self, 
                          reachable_objects: List[str], 
                          state: namo_rl.RLState,
                          env: namo_rl.RLEnvironment) -> Optional[List[str]]:
        """Attempt ML-based object selection."""
        # Load model if needed
        if not self._load_model():
            return None
        
        # Create input data
        json_message = self._create_json_message(reachable_objects, state, env)
        if json_message is None:
            if self.verbose:
                print("Failed to create JSON message for ML inference")
            return None
        
        try:
            # Run inference
            if self.verbose:
                print(f"Running object inference with {len(reachable_objects)} objects")
            
            result = self._object_model.infer(
                json_message=json_message,
                xml_path=json_message["xml_path"],
                robot_goal=json_message["robot_goal"],
                samples=self.samples
            )
            
            # Validate result
            if result['total_valid_samples'] < self.min_valid_samples:
                if self.verbose:
                    print(f"Too few valid samples: {result['total_valid_samples']} < {self.min_valid_samples}")
                return None
            
            # Get vote distribution
            object_votes = result.get('object_votes', Counter())
            if not object_votes:
                if self.verbose:
                    print("No object votes received from ML model")
                return None
            
            # Sort objects by vote count (descending)
            sorted_objects = []
            for obj_id, votes in object_votes.most_common():
                if obj_id in reachable_objects:  # Only include actually reachable objects
                    sorted_objects.append(obj_id)
            
            # Add any reachable objects not voted for (shouldn't happen but be safe)
            for obj_id in reachable_objects:
                if obj_id not in sorted_objects:
                    sorted_objects.append(obj_id)
            
            if self.verbose:
                print(f"ML object ranking: {[(obj, object_votes.get(obj, 0)) for obj in sorted_objects]}")
            
            return sorted_objects
            
        except Exception as e:
            if self.verbose:
                print(f"ML object inference failed: {e}")
            return None
    
    @property
    def strategy_name(self) -> str:
        return "ML Object Selection"


class MLGoalSelectionStrategy(GoalSelectionStrategy):
    """ML-based goal selection using GoalInferenceModel.
    
    Uses trained diffusion models to generate contextually appropriate goals
    for selected objects based on visual scene understanding.
    """
    
    def __init__(self,
                 goal_model_path: str,
                 samples: int = 32,
                 device: str = "cuda",
                 xml_path: Optional[str] = None,
                 min_goals_threshold: int = 1,
                 verbose: bool = False,
                 preloaded_model: Optional[Any] = None,
                 preview_mask_count: int = 0,
                 **unused_kwargs):
        """Initialize ML goal selection strategy.

        Args:
            goal_model_path: Path to trained goal inference model
            samples: Number of diffusion samples for inference
            device: PyTorch device ("cuda" or "cpu")
            xml_path: XML file path (absolute or relative). If None, uses env.get_xml_path()
            min_goals_threshold: Minimum goals before considering result reliable
            verbose: Enable verbose logging
            preloaded_model: Optional pre-loaded model to avoid repeated loading
            preview_mask_count: Number of ML goal masks to preview (0 disables)
            **unused_kwargs: Compatibility placeholder for legacy keyword args
        """
        self.goal_model_path = goal_model_path
        self.samples = samples
        self.device = device
        self.xml_path = xml_path
        self.min_goals_threshold = min_goals_threshold
        self.verbose = verbose
        self.preview_mask_count = max(0, preview_mask_count)
        self._preview_shown = False
        if unused_kwargs and self.verbose:
            print(f"Warning: Unused MLGoalSelectionStrategy kwargs: {list(unused_kwargs.keys())}")
        
        # Use preloaded model if available, otherwise lazy load
        if preloaded_model is not None:
            self._goal_model = preloaded_model
            self._load_attempted = True
            if self.verbose:
                print("Using preloaded GoalInferenceModel")
        else:
            self._goal_model = None
            self._load_attempted = False
    
    def _load_model(self):
        """Lazy load the goal inference model."""
        if self._load_attempted:
            return self._goal_model is not None
        
        self._load_attempted = True
        
        print(f"Loading GoalInferenceModel from {self.goal_model_path}...")
        try:
            from ktamp_learning.goal_inference_model import GoalInferenceModel
            
            if self.verbose:
                print(f"Loading GoalInferenceModel from {self.goal_model_path}")
            
            self._goal_model = GoalInferenceModel(
                model_path=self.goal_model_path,
                device=self.device
            )
            
            print("Goal ML model loaded successfully")
            if self.verbose:
                print("Goal ML model loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to load goal ML model: {e}")
            if self.verbose:
                print(f"Failed to load goal ML model: {e}")
            self._goal_model = None
            return False
    
    def generate_goals(self, 
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[Goal]:
        """Generate goals using ML model."""
        # Try ML inference
        ml_result = self._generate_goals_ml(object_id, state, env, max_goals)
        if ml_result:
            return ml_result[:max_goals]  # Ensure we don't exceed max_goals
        
        # If ML fails, return empty list (will cause this action to be skipped)
        if self.verbose:
            print("ML goal generation failed, skipping this object")
        return []
    
    def _generate_goals_ml(self,
                          object_id: str,
                          state: namo_rl.RLState,
                          env: namo_rl.RLEnvironment,
                          max_goals: int) -> Optional[List[Goal]]:
        """Attempt ML-based goal generation."""
        # Load model if needed
        if not self._load_model():
            print(f"❌ ML model loading failed for {object_id}")
            return None

        # Create input data for goal generation
        if self.verbose:
            print(f"📝 Creating JSON message for {object_id}...")
        json_message = self._create_json_message_for_goals(object_id, state, env)
        if json_message is None:
            print(f"❌ Failed to create JSON message for {object_id}")
            return None

        if self.verbose:
            print(f"✅ JSON message created for {object_id}, calling infer()...")
            print(f"   Using XML path: {json_message.get('xml_path', 'MISSING')}")

        try:
            # Run goal model directly with new independent API
            if self.verbose:
                print(f"Running goal inference for object {object_id} with {self.samples} samples")
            
            goals = self._goal_model.infer(
                json_message=json_message,
                xml_path=json_message["xml_path"],
                robot_goal=json_message["robot_goal"],
                selected_object=object_id,
                samples=self.samples
            )

            if self.verbose:
                print(f"🔍 ML INFERENCE RESULT for {object_id}: {len(goals) if goals else 0} goals generated")
                if goals:
                    for i, g in enumerate(goals[:5]):
                        print(f"  Goal {i}: x={g.get('x', 0):.3f}, y={g.get('y', 0):.3f}, theta={g.get('theta', 0):.3f}")
            else:
                 # Concise summary in non-verbose mode
                 pass 

            if not goals or len(goals) < self.min_goals_threshold:
                if self.verbose:
                    print(f"Too few goals generated: {len(goals)} < {self.min_goals_threshold}")
                return None
            
            self._preview_goal_inference(goals, object_id)
            
            # Convert to our Goal format
            converted_goals = []
            for goal_data in goals:
                if 'x' in goal_data and 'y' in goal_data and 'theta' in goal_data:
                    converted_goals.append(Goal(
                        x=float(goal_data['x']),
                        y=float(goal_data['y']),
                        theta=float(goal_data['theta'])
                    ))
            
            if self.verbose:
                print(f"Generated {len(converted_goals)} ML goals for {object_id}")
            
            return converted_goals
            
        except Exception as e:
            print(f"❌ ML goal inference failed for {object_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _preview_goal_inference(self, goals: List[Dict[str, Any]], object_id: str):
        """Display ML input channels and goal mask predictions using matplotlib."""
        if self.preview_mask_count <= 0 or self._preview_shown:
            return

        try:
            import numpy as np
        except Exception as e:
            if self.verbose:
                print(f"Unable to import numpy for mask preview: {e}")
            self.preview_mask_count = 0
            return

        mask_entries = []
        input_channels = None

        for goal in goals:
            sample = goal.get('goal_sample')
            if sample is None:
                continue
            sample_np = np.asarray(sample)
            if sample_np.ndim == 3:
                mask = sample_np[:, :, 0]
            elif sample_np.ndim == 2:
                mask = sample_np
            else:
                continue

            # Get input channels from first goal (same for all goals in this batch)
            if input_channels is None and 'input_channels' in goal:
                input_channels = goal['input_channels']

            mask_entries.append((goal.get('index', len(mask_entries)), mask))

        if not mask_entries:
            return

        count = min(self.preview_mask_count, len(mask_entries))

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            if self.verbose:
                print(f"Matplotlib not available for goal mask preview: {e}")
            self.preview_mask_count = 0
            return

        try:
            # Create figure with better layout: Input on left (larger), Goal Grid on right
            count = min(self.preview_mask_count, len(mask_entries))
            
            # Calculate grid dimensions for goals
            cols = int(np.ceil(np.sqrt(count)))
            rows = int(np.ceil(count / cols))
            
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])  # Split 50/50 left/right
            
            # Left side: Input visualization
            if input_channels is not None:
                # Denormalize from [-1, 1] to [0, 1]
                inp = (input_channels + 1) / 2

                # Create combined scene visualization
                ax_input = fig.add_subplot(gs[0])

                # Combine channels: robot (0) + goal (1) + movable (2) + static (3)
                combined_scene = np.clip(
                    inp[0, :, :] +      # robot
                    inp[1, :, :] +      # robot goal
                    inp[2, :, :] +      # movable objects
                    inp[3, :, :],       # static objects
                    0, 1
                )

                # Determine selected object mask channel
                # Typically at index 5 if available, otherwise fallback or use index 4 if size allows
                selected_mask_idx = 5
                if selected_mask_idx >= inp.shape[0]:
                    # Fallback: try index 4 (sometimes used for selection if 5-channel input)
                    selected_mask_idx = 4
                
                if selected_mask_idx < inp.shape[0]:
                    selected_mask = inp[selected_mask_idx, :, :]
                else:
                    # No mask channel available
                    selected_mask = np.zeros_like(combined_scene)

                # Create RGB visualization: scene in grayscale, selected object in red
                rgb_img = np.stack([
                    combined_scene + selected_mask,  # Red channel: scene + selected object
                    combined_scene,                   # Green channel: just scene
                    combined_scene                    # Blue channel: just scene
                ], axis=-1)
                rgb_img = np.clip(rgb_img, 0, 1)

                ax_input.imshow(rgb_img)
                ax_input.set_title(f"Input: Scene + Selected Object ({object_id}) in Red", fontsize=14, fontweight='bold')
                ax_input.axis('off')

            # Right side: Grid of Goal Predictions
            # Create a sub-gridspec for the right panel
            gs_right = gs[1].subgridspec(rows, cols, wspace=0.1, hspace=0.1)
            
            for i, (idx, mask) in enumerate(mask_entries[:count]):
                r, c = divmod(i, cols)
                ax = fig.add_subplot(gs_right[r, c])
                ax.imshow(mask, cmap='viridis')
                # ax.set_title(f"Goal #{idx}", fontsize=8)
                ax.axis('off')
                # Add border to make separate plots distinct
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(0.5)

            fig.suptitle(f"ML Goal Inference: {object_id} (Top {count} predictions)", fontsize=16, fontweight='bold', y=0.98)
            # fig.tight_layout() # Removed tight_layout as it can mess with custom gs
            print("🖼️ Close the ML goal visualization window to continue planning...")
            plt.show(block=True)
        except Exception as e:
            if self.verbose:
                print(f"Failed to render ML goal mask preview: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._preview_shown = True
    
    def _create_json_message_for_goals(self,
                                     object_id: str,
                                     state: namo_rl.RLState,
                                     env: namo_rl.RLEnvironment) -> Optional[Dict[str, Any]]:
        """Create JSON message format expected by GoalInferenceModel."""
        # Save original state to restore later
        original_state = env.get_full_state()

        try:
            # Set state to get observations
            env.set_full_state(state)
            obs = env.get_observation()

            # Get robot position and goal
            robot_pose = obs.get('robot_pose')
            if robot_pose is None or len(robot_pose) < 3:
                print(f"  ⚠️ JSON creation failed: No valid robot_pose in observations")
                return None

            # Get robot goal
            robot_goal = None
            if hasattr(env, '_robot_goal'):
                robot_goal = getattr(env, '_robot_goal')
            elif 'robot_goal' in obs:
                robot_goal = obs['robot_goal']
            elif hasattr(env, 'get_robot_goal'):
                try:
                    robot_goal = env.get_robot_goal()
                except:
                    pass

            if robot_goal is None or len(robot_goal) < 2:
                print(f"  ⚠️ JSON creation failed: No valid robot_goal (robot_goal={robot_goal})")
                return None
            
            # Get reachable objects from environment
            reachable_objects = []
            try:
                reachable_objects = env.get_reachable_objects(state)
                if reachable_objects is None:
                    reachable_objects = []
            except:
                reachable_objects = []
            
            # Build objects dictionary - include all objects visible, not just target
            objects_dict = {}
            for key, value in obs.items():
                if key.endswith('_pose') and key != 'robot_pose':
                    obj_name = key[:-5]  # Remove '_pose' suffix
                    if len(value) >= 3:
                        # Convert theta angle (value[2]) to quaternion
                        from scipy.spatial.transform import Rotation as R
                        import numpy as np
                        quat = R.from_euler('xyz', [0, 0, value[2]], degrees=False).as_quat(scalar_first=True)
                        objects_dict[obj_name] = {
                            "position": [float(value[0]), float(value[1]), float(value[2])],
                            "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                        }
            
            # Add static objects from get_object_info()
            try:
                static_info = env.get_object_info()
                # print(f"DEBUG: Found {len(static_info)} static objects: {list(static_info.keys())}")
                for obj_name, obj_data in static_info.items():
                    # print(f"DEBUG: Static object {obj_name}: {obj_data}")
                    if obj_name not in objects_dict:  # Don't override movable objects
                        # Only add objects that have position data and are actually static walls
                        pos_x = obj_data.get('pos_x', None)
                        pos_y = obj_data.get('pos_y', None)
                        
                        # Skip objects without position data (like movable objects that only have size info)
                        # Also skip the robot since it's not a static obstacle
                        if pos_x is None or pos_y is None or obj_name == 'robot':
                            continue
                            
                        angle_deg = obj_data.get('angle_deg', 0.0)
                        
                        # print(f"DEBUG: Adding static object {obj_name} at ({pos_x}, {pos_y}) with angle {angle_deg}°")
                        
                        # Convert angle from degrees to quaternion
                        quat = R.from_euler('xyz', [0, 0, angle_deg], degrees=True).as_quat(scalar_first=True)
                        objects_dict[obj_name] = {
                            "position": [float(pos_x), float(pos_y), float(angle_deg * np.pi / 180.0)],  # Keep theta in position for compatibility
                            "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                        }
            except Exception as e:
                # print(f"DEBUG: Failed to get static objects: {e}")
                # If get_object_info fails, continue without static objects
                pass
            
            # Ensure target object is included
            if object_id not in objects_dict:
                pose_key = f"{object_id}_pose"
                if pose_key in obs and len(obs[pose_key]) >= 3:
                    pose = obs[pose_key]
                    objects_dict[object_id] = {
                        "position": [float(pose[0]), float(pose[1]), float(pose[2])],
                        "quaternion": [1.0, 0.0, 0.0, 0.0]
                    }
                else:
                    return None  # Can't find target object
            
            # Get XML path - use provided path or get from environment
            xml_path = self.xml_path
            if xml_path is None:
                # Try to get from environment
                if hasattr(env, 'get_xml_path'):
                    xml_path = env.get_xml_path()
                else:
                    print(f"  ⚠️ JSON creation failed: No XML path provided and env.get_xml_path() not available")
                    return None
            
            # Create message in expected format
            json_message = {
                "xml_path": xml_path,
                "robot_goal": [float(robot_goal[0]), float(robot_goal[1])],
                "reachable_objects": reachable_objects,
                "robot": {
                    "position": [float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])]
                },
                "objects": objects_dict
            }
            
            return json_message
            
        finally:
            # Always restore original state
            env.set_full_state(original_state)
    
    @property
    def strategy_name(self) -> str:
        return "ML Goal Generation"


class EpsilonGreedyGoalStrategy(GoalSelectionStrategy):
    """Epsilon-greedy goal selection that mixes ML and random strategies.
    
    For each goal slot, randomly chooses between ML and random strategy
    based on epsilon probability. This provides exploration vs exploitation
    balance in goal generation.
    """
    
    def __init__(self,
                 ml_strategy: MLGoalSelectionStrategy,
                 random_strategy: RandomGoalStrategy,
                 epsilon: float = 0.1,
                 verbose: bool = False):
        """Initialize epsilon-greedy goal selection strategy.
        
        Args:
            ml_strategy: ML-based goal selection strategy
            random_strategy: Random goal selection strategy
            epsilon: Probability of selecting random goal (0.0 = pure ML, 1.0 = pure random)
            verbose: Enable verbose logging
        """
        self.ml_strategy = ml_strategy
        self.random_strategy = random_strategy
        self.epsilon = epsilon
        self.verbose = verbose
        
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"Epsilon must be between 0.0 and 1.0, got {epsilon}")
    
    def generate_goals(self, 
                      object_id: str,
                      state: namo_rl.RLState,
                      env: namo_rl.RLEnvironment,
                      max_goals: int) -> List[Goal]:
        """Generate goals using epsilon-greedy selection between ML and random."""
        if max_goals <= 0:
            return []
        
        # Generate full sets from both strategies
        ml_goals = []
        random_goals = []
        
        # Try ML strategy first
        try:
            ml_goals = self.ml_strategy.generate_goals(object_id, state, env, max_goals)
            if self.verbose:
                print(f"ML strategy generated {len(ml_goals)} goals for {object_id}")
        except Exception as e:
            if self.verbose:
                print(f"ML strategy failed for {object_id}: {e}")
        
        # Generate random goals
        try:
            random_goals = self.random_strategy.generate_goals(object_id, state, env, max_goals)
            if self.verbose:
                print(f"Random strategy generated {len(random_goals)} goals for {object_id}")
        except Exception as e:
            if self.verbose:
                print(f"Random strategy failed for {object_id}: {e}")
        
        # If both strategies failed, return empty list
        if not ml_goals and not random_goals:
            if self.verbose:
                print(f"Both strategies failed for {object_id}")
            return []
        
        # Mix goals based on epsilon probability
        final_goals = []
        
        for i in range(max_goals):
            use_random = random.random() < self.epsilon
            
            if use_random and i < len(random_goals):
                # Use random goal
                final_goals.append(random_goals[i])
                if self.verbose:
                    print(f"Goal {i+1}: Selected random goal")
            elif not use_random and i < len(ml_goals):
                # Use ML goal
                final_goals.append(ml_goals[i])
                if self.verbose:
                    print(f"Goal {i+1}: Selected ML goal")
            elif i < len(ml_goals):
                # Fallback to ML if random not available
                final_goals.append(ml_goals[i])
                if self.verbose:
                    print(f"Goal {i+1}: Fallback to ML goal")
            elif i < len(random_goals):
                # Fallback to random if ML not available
                final_goals.append(random_goals[i])
                if self.verbose:
                    print(f"Goal {i+1}: Fallback to random goal")
            else:
                # No more goals available from either strategy
                break
        
        if self.verbose:
            print(f"Final mixed goals for {object_id}: {len(final_goals)} goals (epsilon={self.epsilon})")
        
        return final_goals
    
    @property
    def strategy_name(self) -> str:
        return f"Epsilon-Greedy Goal Generation (ε={self.epsilon})"