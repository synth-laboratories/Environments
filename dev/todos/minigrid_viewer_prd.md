# MiniGrid Viewer Implementation PRD

## Problem Statement

The MiniGrid environment currently lacks a dedicated trace viewer, limiting analysis capabilities for this important grid-based environment. MiniGrid is widely used for testing navigation, goal-reaching, and instruction-following capabilities, but agents' decision-making processes are difficult to analyze without proper visualization.

## Current State Analysis

### ✅ **Existing Infrastructure**
- Environment detection framework in `streamlit_app.py`
- Two-column layout pattern from other environments
- Basic trace processing structure
- Database integration for trace storage

### ❌ **Missing Components**
- No MiniGrid-specific trace processor
- No grid state visualization
- No action mapping for MiniGrid actions
- No integration with main viewer

## Technical Requirements

### 1. **Grid State Visualization**
**Priority: High** | **Effort: 6-8 hours**

**Required Implementation:**
```python
def render_minigrid_grid(grid_state: np.ndarray, agent_pos: Tuple[int, int], agent_dir: int) -> str:
    """Convert MiniGrid state to ASCII representation."""
    if grid_state is None:
        return "No grid data available"
    
    # MiniGrid object type mappings
    object_map = {
        0: "?",  # unseen
        1: ".",  # empty
        2: "#",  # wall
        3: "F",  # floor
        4: "D",  # door (closed)
        5: "d",  # door (open)
        6: "K",  # key
        7: "B",  # ball
        8: "X",  # box
        9: "G",  # goal
        10: "L", # lava
        11: "A"  # agent (fallback)
    }
    
    # Direction symbols for agent
    direction_symbols = ["→", "↓", "←", "↑"]
    
    # Build grid visualization
    height, width = grid_state.shape[:2]
    grid_lines = []
    
    for y in range(height):
        line = []
        for x in range(width):
            if agent_pos and (x, y) == agent_pos:
                # Show agent with direction
                agent_symbol = direction_symbols[agent_dir] if agent_dir is not None else "@"
                line.append(agent_symbol)
            else:
                # Show object at this position
                obj_type = grid_state[y, x, 0] if len(grid_state.shape) > 2 else grid_state[y, x]
                line.append(object_map.get(obj_type, "?"))
        grid_lines.append(" ".join(line))
    
    return "\n".join(grid_lines)
```

### 2. **Action Mapping**
**Priority: High** | **Effort: 1-2 hours**

**Required Implementation:**
```python
def get_minigrid_action_name(action_idx: int) -> str:
    """Map MiniGrid action index to meaningful name."""
    action_names = {
        0: "turn_left",
        1: "turn_right",
        2: "move_forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done"
    }
    return action_names.get(action_idx, f"unknown_action_{action_idx}")
```

### 3. **Trace Processor**
**Priority: High** | **Effort: 8-10 hours**

**Required Implementation:**
```python
def process_minigrid_trace(trace_data: Dict) -> Dict[str, Any]:
    """Process MiniGrid trace data for visualization."""
    trace = trace_data.get("trace", {})
    dataset = trace_data.get("dataset", {})
    metadata = trace.get("metadata", {})
    
    # Extract MiniGrid-specific metadata
    processed = {
        "model_name": metadata.get("model_name", "Unknown"),
        "difficulty": metadata.get("difficulty", "Unknown"),
        "total_reward": dataset.get("reward_signals", [{}])[0].get("reward", 0.0),
        "mission": metadata.get("mission", "Unknown"),
        "grid_size": metadata.get("grid_size", "Unknown"),
        "success": metadata.get("success", False),
        "efficiency_ratio": metadata.get("efficiency_ratio", 0.0),
        "total_steps": metadata.get("total_steps", 0),
        "turns": []
    }
    
    # Process each turn
    partitions = trace.get("partition", [])
    for i, partition in enumerate(partitions[:50]):  # Limit for performance
        events = partition.get("events", [])
        if not events:
            continue
            
        event = events[0]
        env_steps = event.get("environment_compute_steps", [])
        
        # Extract turn data
        turn_data = {
            "turn_number": i + 1,
            "agent_pos": None,
            "agent_dir": None,
            "grid_state": None,
            "carrying": None,
            "action_taken": None,
            "step_count": 0,
            "reward": 0.0,
            "done": False,
            "mission_success": False
        }
        
        # Process environment steps
        for step in env_steps:
            outputs = step.get("compute_output", [{}])[0].get("outputs", {})
            
            # Extract grid state
            if "observation" in outputs:
                obs = outputs["observation"]
                if "image" in obs:
                    turn_data["grid_state"] = obs["image"]
                if "agent_pos" in obs:
                    turn_data["agent_pos"] = tuple(obs["agent_pos"])
                if "agent_dir" in obs:
                    turn_data["agent_dir"] = obs["agent_dir"]
                if "carrying" in obs:
                    turn_data["carrying"] = obs["carrying"]
                if "step_count" in obs:
                    turn_data["step_count"] = obs["step_count"]
            
            # Extract action
            if "action" in outputs:
                action = outputs["action"]
                if isinstance(action, int):
                    turn_data["action_taken"] = get_minigrid_action_name(action)
                else:
                    turn_data["action_taken"] = str(action)
            
            # Extract reward and termination
            if "reward" in outputs:
                turn_data["reward"] = outputs["reward"]
            if "done" in outputs:
                turn_data["done"] = outputs["done"]
            if "info" in outputs:
                info = outputs["info"]
                if "success" in info:
                    turn_data["mission_success"] = info["success"]
        
        processed["turns"].append(turn_data)
    
    return processed
```

### 4. **Turn Renderer**
**Priority: High** | **Effort: 4-6 hours**

**Required Implementation:**
```python
def render_minigrid_turn(turn_data: Dict):
    """Render individual MiniGrid turn."""
    st.subheader(f"Turn {turn_data['turn_number']}")
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Grid State")
        
        # ASCII grid display
        if turn_data["grid_state"] is not None:
            grid_display = render_minigrid_grid(
                turn_data["grid_state"],
                turn_data["agent_pos"],
                turn_data["agent_dir"]
            )
            st.code(grid_display, language=None)
        else:
            st.info("No grid data available")
        
        # Legend
        st.caption("Legend: # = wall, . = empty, G = goal, K = key, D = door, B = ball, L = lava")
        st.caption("Agent: → = right, ↓ = down, ← = left, ↑ = up")
        
        # Action display
        if turn_data["action_taken"]:
            st.write(f"**Action:** {turn_data['action_taken']}")
    
    with col2:
        st.subheader("📊 Agent State")
        
        # Position and direction
        if turn_data["agent_pos"]:
            st.metric("Position", f"({turn_data['agent_pos'][0]}, {turn_data['agent_pos'][1]})")
        
        if turn_data["agent_dir"] is not None:
            direction_names = ["Right", "Down", "Left", "Up"]
            st.metric("Direction", direction_names[turn_data["agent_dir"]])
        
        # Step count and reward
        st.metric("Steps", turn_data["step_count"])
        st.metric("Reward", f"{turn_data['reward']:.3f}")
        
        # Mission status
        if turn_data["mission_success"]:
            st.success("✅ Mission Complete!")
        elif turn_data["done"]:
            st.error("❌ Episode Ended")
        
        # Carrying object
        if turn_data["carrying"]:
            if isinstance(turn_data["carrying"], dict):
                obj_type = turn_data["carrying"].get("type", "unknown")
                obj_color = turn_data["carrying"].get("color", "")
                st.info(f"Carrying: {obj_color} {obj_type}")
            else:
                st.info(f"Carrying: {turn_data['carrying']}")
        else:
            st.info("Not carrying anything")
```

### 5. **Main Trace Renderer**
**Priority: High** | **Effort: 3-4 hours**

**Required Implementation:**
```python
def render_minigrid_trace(processed_trace: Dict):
    """Render MiniGrid trace visualization."""
    # Header
    st.header(f"🎯 MiniGrid Trace - {processed_trace['model_name']}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mission", processed_trace.get("mission", "Unknown"))
    with col2:
        st.metric("Grid Size", processed_trace.get("grid_size", "Unknown"))
    with col3:
        success_icon = "✅" if processed_trace.get("success", False) else "❌"
        st.metric("Success", success_icon)
    with col4:
        st.metric("Total Reward", f"{processed_trace['total_reward']:.3f}")
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Total Steps", processed_trace.get("total_steps", 0))
    with col6:
        st.metric("Efficiency", f"{processed_trace.get('efficiency_ratio', 0.0):.2f}")
    with col7:
        st.metric("Total Turns", len(processed_trace['turns']))
    with col8:
        avg_reward = processed_trace['total_reward'] / max(len(processed_trace['turns']), 1)
        st.metric("Avg Reward/Turn", f"{avg_reward:.3f}")
    
    st.divider()
    
    # Turn selector
    if len(processed_trace['turns']) > 1:
        selected_turn_idx = st.selectbox(
            "Select Turn to View",
            range(len(processed_trace['turns'])),
            format_func=lambda i: f"Turn {i + 1}"
        )
    else:
        selected_turn_idx = 0
    
    # Display selected turn
    if 0 <= selected_turn_idx < len(processed_trace['turns']):
        render_minigrid_turn(processed_trace['turns'][selected_turn_idx])
    
    # Compact view option
    if st.checkbox("Show All Turns (Compact View)"):
        st.subheader("All Turns Overview")
        for i, turn in enumerate(processed_trace['turns'][:20]):  # Limit for performance
            with st.expander(f"Turn {turn['turn_number']}", expanded=False):
                render_minigrid_turn_compact(turn)
```

### 6. **Compact Turn Renderer**
**Priority: Medium** | **Effort: 2-3 hours**

**Required Implementation:**
```python
def render_minigrid_turn_compact(turn_data: Dict):
    """Render compact view of MiniGrid turn."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if turn_data["agent_pos"]:
            st.write(f"**Pos:** {turn_data['agent_pos']}")
        if turn_data["action_taken"]:
            st.write(f"**Action:** {turn_data['action_taken']}")
    
    with col2:
        st.write(f"**Steps:** {turn_data['step_count']}")
        st.write(f"**Reward:** {turn_data['reward']:.3f}")
    
    with col3:
        if turn_data["mission_success"]:
            st.success("✅ Success")
        elif turn_data["done"]:
            st.error("❌ Done")
        else:
            st.info("🔄 Ongoing")
```

## Implementation Plan

### **Phase 1: Core Functionality (Week 1)**
**Goal: Basic MiniGrid visualization working**

1. **Day 1-2: Grid Visualization**
   - Implement `render_minigrid_grid()` function
   - Test with sample grid states
   - Handle different grid sizes and object types

2. **Day 3: Action Mapping**
   - Implement `get_minigrid_action_name()` function
   - Test action display
   - Verify action names match MiniGrid conventions

3. **Day 4-5: Trace Processor**
   - Implement `process_minigrid_trace()` function
   - Test with sample MiniGrid traces
   - Handle edge cases and missing data

### **Phase 2: UI Implementation (Week 2)**
**Goal: Complete MiniGrid viewer interface**

1. **Day 1-2: Turn Renderer**
   - Implement `render_minigrid_turn()` function
   - Test grid display in Streamlit
   - Optimize layout and styling

2. **Day 3: Main Trace Renderer**
   - Implement `render_minigrid_trace()` function
   - Add summary metrics display
   - Implement turn selection

3. **Day 4-5: Integration & Polish**
   - Add MiniGrid detection to main viewer
   - Implement compact view
   - Add error handling and edge cases

### **Phase 3: Testing & Documentation (Week 3)**
**Goal: Production-ready implementation**

1. **Day 1-2: Testing**
   - Unit tests for all functions
   - Integration tests with real traces
   - Performance testing with large traces

2. **Day 3: Documentation**
   - Update README with MiniGrid features
   - Add developer documentation
   - Create usage examples

3. **Day 4-5: Optimization**
   - Performance improvements
   - Memory usage optimization
   - Error handling improvements

## Integration Requirements

### **Main Viewer Integration**
**File: `streamlit_app.py`**

1. **Environment Detection:**
```python
# Add to render_trace_visualization()
if "minigrid" in meta_env_name.lower():
    env_type = "minigrid"
```

2. **Trace Processing:**
```python
# Add to processing section
if env_type == "minigrid":
    processed_trace = process_minigrid_trace(trace_data)
```

3. **Trace Rendering:**
```python
# Add to rendering section
if env_type == "minigrid":
    render_minigrid_trace(processed_trace)
```

### **Database Integration**
Ensure MiniGrid evaluations are properly imported and accessible through existing database queries.

## Testing Strategy

### **Unit Tests**
1. **Grid Rendering Tests**
   - Test various grid sizes (5x5, 7x7, 19x19)
   - Test different object types and combinations
   - Test agent positioning and direction display
   - Test edge cases (empty grids, invalid data)

2. **Action Mapping Tests**
   - Test all MiniGrid action indices (0-6)
   - Test unknown action handling
   - Test action name consistency

3. **Trace Processing Tests**
   - Test with real MiniGrid evaluation traces
   - Test with different MiniGrid environments
   - Test performance with large traces
   - Test error handling with malformed data

### **Integration Tests**
1. **Viewer Integration Tests**
   - Test environment detection
   - Test trace loading and processing
   - Test UI rendering in Streamlit
   - Test navigation between turns

2. **Performance Tests**
   - Test with traces up to 1000 turns
   - Test memory usage with large grids
   - Test rendering performance

### **User Acceptance Tests**
1. **Functionality Tests**
   - Verify grid visualization accuracy
   - Verify action display correctness
   - Verify metrics calculation
   - Verify mission success detection

2. **Usability Tests**
   - Test ease of navigation
   - Test information clarity
   - Test visual appeal and layout

## Success Metrics

### **Functional Requirements**
- ✅ Grid states render correctly as ASCII
- ✅ Agent position and direction display accurately
- ✅ Actions display with meaningful names
- ✅ Mission success/failure clearly indicated
- ✅ All MiniGrid environments supported
- ✅ Performance acceptable for large traces

### **Quality Requirements**
- ✅ Grid visualization covers all MiniGrid object types
- ✅ Action mapping covers all MiniGrid actions
- ✅ UI layout consistent with other environment viewers
- ✅ Error handling for malformed trace data
- ✅ Performance acceptable for traces up to 1000 turns

### **User Experience Requirements**
- ✅ Intuitive navigation between turns
- ✅ Clear visual indicators for mission status
- ✅ Helpful legends and explanations
- ✅ Responsive layout on different screen sizes

## Risk Mitigation

### **Technical Risks**
1. **Grid Size Variations**
   - *Risk*: Different MiniGrid environments use different grid sizes
   - *Mitigation*: Dynamic grid rendering, test with various sizes

2. **Object Type Variations**
   - *Risk*: New MiniGrid environments might introduce new object types
   - *Mitigation*: Fallback to "?" for unknown objects, easy to extend mapping

3. **Performance Issues**
   - *Risk*: Large grids might render slowly
   - *Mitigation*: Limit grid size display, optimize ASCII rendering

### **Integration Risks**
1. **Trace Format Changes**
   - *Risk*: MiniGrid trace format might change
   - *Mitigation*: Version detection, backward compatibility

2. **Database Schema Changes**
   - *Risk*: Changes to evaluation storage format
   - *Mitigation*: Use existing database interface, minimal coupling

## Future Enhancements

### **Phase 4: Advanced Features**
- **Interactive Grid**: Click on grid cells for detailed information
- **Path Visualization**: Show agent movement history
- **Goal Analysis**: Analyze goal-reaching strategies
- **Instruction Following**: Display and analyze language instructions
- **Multi-Agent Support**: Handle multi-agent MiniGrid environments

### **Phase 5: Analysis Tools**
- **Efficiency Analysis**: Calculate optimal vs actual path lengths
- **Strategy Analysis**: Identify common navigation patterns
- **Failure Analysis**: Categorize and analyze mission failures
- **Comparative Analysis**: Compare different agent approaches
- **Learning Curves**: Track improvement over multiple episodes

## Estimated Timeline

### **Development: 3 weeks**
- Week 1: Core functionality (grid rendering, action mapping, trace processing)
- Week 2: UI implementation (turn renderer, main renderer, integration)
- Week 3: Testing, documentation, optimization

### **Testing: 1 week**
- Unit testing: 2-3 days
- Integration testing: 2-3 days
- Performance testing: 1-2 days

### **Documentation: 2-3 days**
- Update README with MiniGrid features
- Add developer documentation
- Create user guide for MiniGrid analysis

**Total Estimated Time: 4-5 weeks**

This implementation will provide comprehensive MiniGrid trace visualization, enabling detailed analysis of agent navigation, goal-reaching, and instruction-following behaviors in grid-based environments. 