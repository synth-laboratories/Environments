# NetHack Viewer Implementation PRD

## Problem Statement

The NetHack trace viewer is currently incomplete with simplified implementations that limit analysis capabilities. Key issues:

1. **Glyph Rendering**: Currently shows placeholder text instead of ASCII dungeon map
2. **Action Mapping**: Uses generic `action_{idx}` instead of meaningful action names
3. **Stats Display**: Shows only basic stats, missing important NetHack attributes
4. **Message Handling**: Limited message display and context

## Current State Analysis

### ✅ **Working Components**
- Basic trace processing structure
- Environment detection (`nethack` in environment name)
- Core stats extraction (HP, level, gold, AC)
- Message extraction from trace data
- Two-column layout framework

### ❌ **Missing/Incomplete Components**
```python
# In frontend/viewers/nethack.py:203
return f"Map: {len(glyphs)}x{len(glyphs[0])} (glyph data not rendered)"

# In frontend/viewers/base.py:272  
return f"action_{action_idx}"  # Simplified NetHack action mapping
```

## Technical Requirements

### 1. **ASCII Map Rendering**
**Priority: High** | **Effort: 6-8 hours**

**Current Issue:**
```python
def format_nethack_glyphs(glyphs: List[List[int]]) -> str:
    if not glyphs:
        return "No map data available"
    return f"Map: {len(glyphs)}x{len(glyphs[0])} (glyph data not rendered)"
```

**Required Implementation:**
```python
def format_nethack_glyphs(glyphs: List[List[int]]) -> str:
    """Convert NetHack glyph array to ASCII representation."""
    if not glyphs:
        return "No map data available"
    
    # Core glyph mappings (simplified but functional)
    glyph_map = {
        # Terrain
        2359: ".",   # floor
        2360: "#",   # wall
        2361: "|",   # vertical wall
        2362: "-",   # horizontal wall
        2363: "+",   # door
        2364: "^",   # trap
        2365: "<",   # up stairs
        2366: ">",   # down stairs
        
        # Monsters (player and common enemies)
        2378: "@",   # player
        2379: "d",   # dog
        2380: "r",   # rat
        2381: "o",   # orc
        2382: "k",   # kobold
        
        # Items
        2400: "!",   # potion
        2401: "?",   # scroll
        2402: "=",   # ring
        2403: "\"",  # amulet
        2404: "/",   # wand
        2405: "(",   # weapon
        2406: "[",   # armor
        2407: "%",   # food
        2408: "$",   # gold
        
        # Default for unknown glyphs
        0: " ",      # empty space
    }
    
    # Build ASCII representation
    lines = []
    for row in glyphs:
        line = ""
        for glyph in row:
            line += glyph_map.get(glyph, "?")
        lines.append(line)
    
    return "\n".join(lines)
```

### 2. **Action Name Mapping**
**Priority: High** | **Effort: 2-3 hours**

**Current Issue:**
```python
def get_action_name(self, action_idx: int) -> str:
    return f"action_{action_idx}"
```

**Required Implementation:**
```python
def get_nethack_action_name(action_idx: int) -> str:
    """Map NetHack action index to meaningful name."""
    action_names = {
        # Movement
        0: "north",
        1: "northeast", 
        2: "east",
        3: "southeast",
        4: "south",
        5: "southwest",
        6: "west",
        7: "northwest",
        8: "up",
        9: "down",
        
        # Interaction
        10: "wait",
        11: "pickup",
        12: "drop",
        13: "eat",
        14: "search",
        15: "open",
        16: "close",
        17: "kick",
        18: "apply",
        19: "read",
        20: "zap",
        21: "throw",
        22: "wear",
        23: "take_off",
        24: "put_on",
        25: "remove",
        
        # Combat
        26: "fight",
        27: "fire",
        
        # Magic/Special
        28: "pray",
        29: "cast_spell",
        30: "invoke",
        31: "offer",
        
        # Navigation
        32: "look",
        33: "teleport",
        34: "jump",
        35: "monster_ability",
        36: "turn_undead",
        37: "untrap",
        
        # Meta
        38: "rest",
        39: "save",
        40: "quit",
        41: "redraw",
        42: "version",
        43: "inventory",
        44: "discoveries",
        45: "list_spells",
        46: "adjust_inventory",
        47: "name_item",
        48: "name_creature",
        49: "name_object_type",
        50: "call_monster",
        51: "call_object",
        52: "macro",
        53: "rush",
        54: "run",
        55: "run2",
        56: "fight2",
        57: "ESC"
    }
    
    return action_names.get(action_idx, f"unknown_action_{action_idx}")
```

### 3. **Enhanced Stats Display**
**Priority: Medium** | **Effort: 3-4 hours**

**Current Implementation:**
```python
turn_data["stats"] = {
    "hp": blstats[10], "max_hp": blstats[11],
    "level": blstats[18], "gold": blstats[13], "ac": blstats[17]
}
```

**Required Enhancement:**
```python
def extract_nethack_stats(blstats: List[int]) -> Dict[str, Any]:
    """Extract comprehensive NetHack character stats."""
    return {
        # Core stats (existing)
        "hp": blstats[10],
        "max_hp": blstats[11], 
        "level": blstats[18],
        "gold": blstats[13],
        "ac": blstats[17],
        
        # Attributes (new)
        "strength": blstats[0],
        "dexterity": blstats[1],
        "constitution": blstats[2], 
        "intelligence": blstats[3],
        "wisdom": blstats[4],
        "charisma": blstats[5],
        
        # Game state (new)
        "depth": blstats[19],
        "experience": blstats[20],
        "experience_level": blstats[18],
        "energy": blstats[14],
        "max_energy": blstats[15],
        "hunger": blstats[21],
        "carrying_capacity": blstats[22],
        "dungeon_level": blstats[12],
        "time": blstats[23]
    }
```

### 4. **Improved Message Display**
**Priority: Medium** | **Effort: 2-3 hours**

**Current Implementation:**
```python
"metadata": {
    "messages": messages[-5:],  # Last 5 messages
    "glyphs": glyphs
}
```

**Required Enhancement:**
```python
def format_nethack_messages(messages: List[str]) -> Dict[str, Any]:
    """Format and categorize NetHack messages."""
    if not messages:
        return {"recent": [], "important": [], "warnings": []}
    
    important_keywords = ["You die", "You ascend", "You feel", "You are", "level up"]
    warning_keywords = ["You are slowing down", "You feel weak", "You are confused"]
    
    categorized = {
        "recent": messages[-5:],  # Last 5 messages
        "important": [],
        "warnings": []
    }
    
    for msg in messages:
        if any(keyword in msg for keyword in important_keywords):
            categorized["important"].append(msg)
        elif any(keyword in msg for keyword in warning_keywords):
            categorized["warnings"].append(msg)
    
    return categorized
```

## Implementation Plan

### **Phase 1: Core Functionality (Week 1)**
**Goal: Get basic ASCII rendering working**

1. **Day 1-2: ASCII Map Rendering**
   - Implement basic glyph-to-ASCII mapping
   - Test with sample NetHack traces
   - Handle edge cases (empty maps, invalid glyphs)

2. **Day 3: Action Mapping**
   - Implement comprehensive action name mapping
   - Test action display in trace viewer
   - Verify action names match NetHack conventions

3. **Day 4-5: Integration & Testing**
   - Update `process_nethack_trace()` to use new functions
   - Test with multiple NetHack evaluation runs
   - Fix any rendering issues

### **Phase 2: Enhanced Features (Week 2)**
**Goal: Add comprehensive stats and message handling**

1. **Day 1-2: Enhanced Stats**
   - Implement full stats extraction
   - Update stats display UI
   - Add attribute display section

2. **Day 3: Message Improvements**
   - Implement message categorization
   - Update message display UI
   - Add message filtering/search

3. **Day 4-5: Polish & Documentation**
   - Add comprehensive error handling
   - Update documentation
   - Performance optimization

## UI/UX Design

### **Enhanced Stats Display**
```python
def render_nethack_stats(stats: Dict[str, Any]):
    """Render comprehensive NetHack stats."""
    # Core stats (existing layout)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("HP", f"{stats['hp']}/{stats['max_hp']}")
        st.metric("Level", stats['level'])
    with col2:
        st.metric("Gold", stats['gold'])
        st.metric("AC", stats['ac'])
    with col3:
        st.metric("Depth", stats['depth'])
        st.metric("Energy", f"{stats['energy']}/{stats['max_energy']}")
    
    # Attributes section (new)
    st.subheader("⚔️ Attributes")
    attr_col1, attr_col2, attr_col3 = st.columns(3)
    with attr_col1:
        st.write(f"**STR:** {stats['strength']}")
        st.write(f"**DEX:** {stats['dexterity']}")
    with attr_col2:
        st.write(f"**CON:** {stats['constitution']}")
        st.write(f"**INT:** {stats['intelligence']}")
    with attr_col3:
        st.write(f"**WIS:** {stats['wisdom']}")
        st.write(f"**CHA:** {stats['charisma']}")
```

### **Enhanced Message Display**
```python
def render_nethack_messages(messages: Dict[str, List[str]]):
    """Render categorized NetHack messages."""
    if messages["important"]:
        st.subheader("🚨 Important Messages")
        for msg in messages["important"][-3:]:
            st.success(msg)
    
    if messages["warnings"]:
        st.subheader("⚠️ Warnings")
        for msg in messages["warnings"][-3:]:
            st.warning(msg)
    
    st.subheader("💬 Recent Messages")
    for msg in messages["recent"]:
        st.text(msg)
```

## Testing Strategy

### **Unit Tests**
1. **Glyph Mapping Tests**
   - Test common glyph-to-ASCII conversions
   - Test edge cases (empty maps, invalid glyphs)
   - Test map size variations

2. **Action Mapping Tests**
   - Test all known action indices
   - Test unknown action handling
   - Test action name consistency

3. **Stats Extraction Tests**
   - Test blstats array parsing
   - Test missing/invalid stats handling
   - Test stats display formatting

### **Integration Tests**
1. **Trace Processing Tests**
   - Test with real NetHack evaluation traces
   - Test with different NetHack versions
   - Test performance with large traces

2. **UI Rendering Tests**
   - Test map display in Streamlit
   - Test stats display layout
   - Test message display formatting

## Success Metrics

### **Functional Requirements**
- ✅ ASCII dungeon maps render correctly
- ✅ Action names display meaningfully
- ✅ All character stats visible
- ✅ Messages categorized and displayed
- ✅ No crashes with invalid data

### **Quality Requirements**
- ✅ Glyph mapping covers 95% of common glyphs
- ✅ Action mapping covers all NetHack actions
- ✅ Stats display matches NetHack conventions
- ✅ Performance acceptable for traces up to 1000 turns
- ✅ Error handling for malformed trace data

## Risk Mitigation

### **Technical Risks**
1. **Incomplete Glyph Mapping**
   - *Risk*: NetHack has 6000+ glyphs, mapping all is impractical
   - *Mitigation*: Focus on 200-300 most common glyphs, fallback to "?" for unknown

2. **Action Index Variations**
   - *Risk*: Action indices might vary between NetHack versions
   - *Mitigation*: Include version detection, maintain multiple mappings if needed

3. **Performance Issues**
   - *Risk*: Large ASCII maps might render slowly
   - *Mitigation*: Limit map size display, add pagination for large traces

### **Integration Risks**
1. **Trace Format Changes**
   - *Risk*: NetHack trace format might change
   - *Mitigation*: Add version detection, maintain backward compatibility

2. **UI Layout Issues**
   - *Risk*: ASCII maps might not display well in Streamlit
   - *Mitigation*: Use `st.code()` with monospace font, test on different screen sizes

## Future Enhancements

### **Phase 3: Advanced Features**
- **Interactive Map**: Click on map tiles for detailed information
- **Inventory Display**: Show character inventory and equipment
- **Spell List**: Display known spells and their status
- **Combat Analysis**: Track combat effectiveness and damage
- **Pathfinding Visualization**: Show agent movement patterns

### **Phase 4: Analysis Tools**
- **Death Analysis**: Categorize and analyze character deaths
- **Progress Tracking**: Monitor dungeon exploration progress
- **Strategy Analysis**: Identify common agent strategies
- **Comparative Analysis**: Compare different agent approaches

## Estimated Timeline

### **Development: 2 weeks**
- Week 1: Core functionality (ASCII rendering, action mapping)
- Week 2: Enhanced features (stats, messages, polish)

### **Testing: 3-4 days**
- Unit testing: 1-2 days
- Integration testing: 1-2 days
- Performance testing: 1 day

### **Documentation: 1-2 days**
- Update README with NetHack features
- Add developer documentation
- Create user guide for NetHack analysis

**Total Estimated Time: 3-4 weeks**

This comprehensive implementation will transform the NetHack viewer from a basic placeholder into a fully functional analysis tool that provides deep insights into agent behavior in the complex NetHack environment. 