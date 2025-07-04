"""
Crafter-specific viewer implementation.
"""

import json
import base64
from typing import Dict, Any, List
from .base_v2 import BaseEnvironmentViewer, viewer_registry


class CrafterViewer(BaseEnvironmentViewer):
    """Viewer for Crafter environment with rich visualizations."""
    
    def get_environment_name(self) -> str:
        return "crafter"
    
    def validate_trace_structure(self, trace_data: Dict[str, Any]) -> None:
        """Validate Crafter trace structure."""
        # Check top level
        assert 'trace' in trace_data, "Missing 'trace' key"
        assert 'dataset' in trace_data, "Missing 'dataset' key"
        
        trace = trace_data['trace']
        assert 'metadata' in trace, "Missing 'trace.metadata'"
        assert 'partition' in trace, "Missing 'trace.partition'"
        assert isinstance(trace['partition'], list), "'trace.partition' must be a list"
        assert len(trace['partition']) > 0, "'trace.partition' is empty"
        
        # Check first partition structure
        first_partition = trace['partition'][0]
        assert 'events' in first_partition, "Missing 'events' in partition"
        assert isinstance(first_partition['events'], list), "'events' must be a list"
        assert len(first_partition['events']) > 0, "'events' is empty"
        
        # Check for Crafter-specific data
        first_event = first_partition['events'][0]
        if 'environment_compute_steps' in first_event:
            env_steps = first_event['environment_compute_steps']
            if len(env_steps) > 0 and 'output' in env_steps[0]:
                output = env_steps[0]['output']
                # Crafter should have images in outputs
                assert 'outputs' in output or 'image' in output, "Missing Crafter image data"
    
    def get_trace_summary(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Crafter-specific summary."""
        trace = trace_data['trace']
        dataset = trace_data.get('dataset', {})
        
        # Count turns
        num_turns = len(trace['partition'])
        
        # Get model name
        model_name = trace.get('metadata', {}).get('model_name', 'unknown')
        
        # Get final reward
        reward_signals = dataset.get('reward_signals', [])
        total_reward = reward_signals[0].get('reward', 0.0) if reward_signals else 0.0
        
        # Extract achievements
        achievements = set()
        for partition in trace['partition']:
            for event in partition.get('events', []):
                for env_step in event.get('environment_compute_steps', []):
                    if 'output' in env_step:
                        logs = env_step['output'].get('logs', [])
                        for log in logs:
                            if isinstance(log, dict) and log.get('type') == 'achievement':
                                achievements.add(log.get('name', 'unknown'))
        
        return {
            'num_turns': num_turns,
            'model_name': model_name,
            'total_reward': total_reward,
            'custom_stats': {
                'achievements': list(achievements),
                'num_achievements': len(achievements),
                'difficulty': trace.get('metadata', {}).get('difficulty', 'unknown')
            }
        }
    
    def render_turn_html(self, trace_data: Dict[str, Any], turn_index: int) -> str:
        """Render Crafter turn with available data."""
        try:
            partition = trace_data['trace']['partition'][turn_index]
            event = partition['events'][0]
            
            html = f'<div class="crafter-turn">'
            html += f'<h3>Turn {turn_index + 1}</h3>'
            
            # Extract available data
            actions = []
            images = []
            player_stats = []
            rewards = []
            
            for env_step in event.get('environment_compute_steps', []):
                if 'compute_output' in env_step and env_step['compute_output']:
                    output = env_step['compute_output'][0]
                    outputs = output.get('outputs', {})
                    
                    # Get action
                    if 'action_index' in outputs:
                        action_idx = outputs['action_index']
                        action_name = self._get_action_name(action_idx)
                        actions.append(f"{action_name} ({action_idx})")
                    
                    # Get image - check both 'image' and 'image_base64'
                    if 'image_base64' in outputs:
                        img_data = outputs['image_base64']
                        images.append(f"data:image/png;base64,{img_data}")
                    elif 'image' in outputs:
                        img_data = outputs['image']
                        if img_data.startswith('data:image'):
                            images.append(img_data)
                        else:
                            images.append(f"data:image/png;base64,{img_data}")
                    
                    # Get player stats
                    if 'player_stats' in outputs:
                        stats = outputs['player_stats']
                        player_stats.append(stats)
                    
                    # Get rewards
                    if 'reward' in outputs:
                        rewards.append({
                            'step': outputs.get('reward', 0),
                            'total': outputs.get('total_reward', 0)
                        })
            
            # Render actions
            if actions:
                html += '<div class="crafter-actions">'
                html += f'<strong>Actions:</strong> {", ".join(actions)}'
                html += '</div>'
            
            # Render player stats
            if player_stats:
                stats = player_stats[-1]  # Latest stats
                html += '<div class="crafter-stats">'
                html += '<strong>Player Stats:</strong> '
                html += f'❤️ Health: {stats.get("health", 0)} | '
                html += f'🍖 Food: {stats.get("food", 0)} | '
                html += f'💧 Drink: {stats.get("drink", 0)}'
                html += '</div>'
            
            # Render rewards
            if rewards:
                reward = rewards[-1]  # Latest reward
                html += '<div class="crafter-rewards">'
                html += f'<strong>Reward:</strong> Step: {reward["step"]:.3f}, Total: {reward["total"]:.3f}'
                html += '</div>'
            
            # Render images if available
            if images:
                html += '<div class="crafter-images">'
                for i, img in enumerate(images):
                    html += f'''
                    <div class="crafter-image-container">
                        <img src="{img}" alt="Turn {turn_index + 1} Frame {i + 1}" 
                             class="crafter-image" />
                        <div class="image-caption">Frame {i + 1}</div>
                    </div>
                    '''
                html += '</div>'
            else:
                # Note about missing images
                html += '''
                <div class="info" style="margin-top: 15px;">
                    <strong>Note:</strong> Images are not available in this trace. 
                    The trace may have been recorded without image capture enabled.
                </div>
                '''
            
            # Show raw data for debugging
            html += '''
            <details class="debug-details">
                <summary>🔍 View Raw Data</summary>
                <pre>''' + json.dumps(event, indent=2) + '</pre>'
            html += '</details>'
            
            html += '</div>'
            return html
            
        except Exception as e:
            return f'''
            <div class="error">
                Failed to render turn {turn_index + 1}: {str(e)}
                <pre>{json.dumps(self.extract_turn_metadata(trace_data, turn_index), indent=2)}</pre>
            </div>
            '''
    
    def _get_action_name(self, action_idx: int) -> str:
        """Map Crafter action index to name."""
        action_names = {
            0: "noop",
            1: "move_left", 
            2: "move_right",
            3: "move_up",
            4: "move_down",
            5: "do",
            6: "sleep",
            7: "place_stone",
            8: "place_table",
            9: "place_furnace",
            10: "place_plant",
            11: "make_wood_pickaxe",
            12: "make_stone_pickaxe",
            13: "make_iron_pickaxe",
            14: "make_wood_sword",
            15: "make_stone_sword", 
            16: "make_iron_sword"
        }
        return action_names.get(action_idx, f"unknown_{action_idx}")
    
    def get_custom_css(self) -> str:
        """Crafter-specific CSS."""
        return """
        .crafter-turn {
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .crafter-actions {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
        }
        
        .crafter-images {
            display: flex;
            gap: 15px;
            margin: 15px 0;
            overflow-x: auto;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        
        .crafter-image-container {
            text-align: center;
            flex-shrink: 0;
        }
        
        .crafter-image {
            max-width: 300px;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
            image-rendering: pixelated;
        }
        
        .image-caption {
            margin-top: 5px;
            font-size: 12px;
            color: #666;
        }
        
        .crafter-achievements {
            background: #e8f5e9;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .crafter-stats {
            background: #fff3e0;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
        }
        
        .crafter-rewards {
            background: #f3e5f5;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
        }
        
        .achievement-badge {
            display: inline-block;
            background: #4caf50;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 0 5px;
        }
        
        .debug-details {
            margin-top: 15px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        
        .debug-details summary {
            cursor: pointer;
            font-weight: 500;
            color: #666;
        }
        
        .debug-details pre {
            margin-top: 10px;
            max-height: 300px;
            overflow: auto;
        }
        """
    
    def get_custom_javascript(self) -> str:
        """Crafter-specific JavaScript."""
        return """
        // Crafter-specific interactions
        function toggleDebugDetails(element) {
            const details = element.parentElement;
            details.open = !details.open;
        }
        
        // Image zoom on click
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('crafter-image')) {
                // Could implement a lightbox here
                console.log('Image clicked:', e.target.src);
            }
        });
        """


# Register the Crafter viewer
viewer_registry.register(CrafterViewer())