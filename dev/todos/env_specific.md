# Environment-Specific Plugin Refactor PRD

## Problem Statement

The trace viewer currently hardcodes logic for four environments (Crafter, Sokoban, NetHack, MiniGrid) across multiple files, making it difficult to add new environments without touching core viewer code. This creates maintenance burden and breaks the principle of extensibility.

### Current Pain Points

1. **Hardcoded discovery paths**: Static array of 18+ glob patterns in `streamlit_app.py`
2. **Environment detection via string matching**: Chain of `elif` statements checking path names
3. **Duplicated processing logic**: Four separate `process_*_trace` functions with similar boilerplate
4. **Duplicated rendering logic**: Four separate `render_*_trace` functions with similar patterns
5. **Action mapping scattered**: Hardcoded dictionaries embedded in main file
6. **Registry maintenance**: Manual updates to `PROCESSORS` dict in frontend code

### Impact

- Adding a new environment requires editing 3-4 files across different directories
- Risk of breaking existing environments when adding new ones
- 1900+ line `streamlit_app.py` file that keeps growing
- Duplicated logic that can drift out of sync

## Solution Overview

Create a **plugin architecture** where each environment is self-contained in a single file that implements a standard interface. The core viewer becomes generic and discovers/loads plugins automatically.

### Key Principles

1. **Zero behavioral change** for existing environments
2. **Single file per environment** - all env-specific logic containerized
3. **Automatic discovery** - no manual registry updates
4. **Backward compatibility** - existing traces continue to work
5. **Testable** - comprehensive regression tests to ensure no breakage

## Technical Design

### 1. Plugin Interface

Create `src/viewer/env_plugins/base.py`:

```python
from typing import Protocol, List, Dict, Any, Optional
from pathlib import Path
from abc import abstractmethod

class TraceProcessor(Protocol):
    @abstractmethod
    def process_turns(self, partitions: List[Dict]) -> List[Dict]:
        """Process raw partition data into turn data."""
        pass
    
    @abstractmethod
    def get_action_name(self, action_idx: int) -> str:
        """Map action index to human-readable name."""
        pass

class EnvPlugin(Protocol):
    # REQUIRED ATTRIBUTES
    name: str                           # "crafter", "sokoban", etc.
    display_name: str                   # "Crafter Classic", "Sokoban Puzzles"
    eval_run_globs: List[str]          # Glob patterns to find run_* directories
    
    # REQUIRED METHODS
    @abstractmethod
    def identify(self, run_path: Path, summary: Optional[Dict]) -> bool:
        """Determine if this plugin handles the given evaluation run."""
        pass
    
    @abstractmethod
    def create_processor(self) -> TraceProcessor:
        """Create a trace processor instance."""
        pass
    
    @abstractmethod
    def render_trace(self, st, processed_trace: Dict) -> None:
        """Render the trace visualization using Streamlit."""
        pass
    
    # OPTIONAL METHODS
    def extract_metadata(self, run_path: Path, summary: Dict, trace_data: Dict) -> Dict:
        """Extract environment-specific metadata during import."""
        return {}
    
    def validate_trace(self, trace_data: Dict) -> bool:
        """Validate that trace data is compatible with this environment."""
        return True
```

### 2. Plugin Implementation Structure

Each environment gets `src/viewer/env_plugins/<env_name>.py`:

```python
from .base import EnvPlugin, TraceProcessor
from typing import List, Dict, Any, Optional
from pathlib import Path

class CrafterProcessor(TraceProcessor):
    def process_turns(self, partitions: List[Dict]) -> List[Dict]:
        # Move existing process_crafter_trace logic here
        pass
    
    def get_action_name(self, action_idx: int) -> str:
        # Move existing get_crafter_action_name logic here
        pass

class CrafterPlugin(EnvPlugin):
    name = "crafter"
    display_name = "Crafter Classic"
    eval_run_globs = [
        "../synth_env/examples/crafter_classic/agent_demos/src/evals/crafter/run_*",
        "../src/evals/crafter/run_*",
        "../../src/evals/crafter/run_*",
        "../../../src/evals/crafter/run_*",
    ]
    
    def identify(self, run_path: Path, summary: Optional[Dict]) -> bool:
        # Check explicit environment_name first
        if summary and summary.get('environment_name') == 'crafter':
            return True
        # Fallback to path heuristics
        return 'crafter' in str(run_path).lower()
    
    def create_processor(self) -> TraceProcessor:
        return CrafterProcessor()
    
    def render_trace(self, st, processed_trace: Dict) -> None:
        # Move existing render_crafter_trace logic here
        pass

# Export the plugin instance
plugin = CrafterPlugin()
```

### 3. Registry and Auto-Discovery

In `src/viewer/streamlit_app.py`:

```python
from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
from .env_plugins.base import EnvPlugin

class PluginRegistry:
    def __init__(self):
        self._plugins: Dict[str, EnvPlugin] = {}
        self._discover_plugins()
    
    def _discover_plugins(self):
        """Automatically discover and load all plugins."""
        plugin_dir = Path(__file__).parent / "env_plugins"
        
        for module_info in iter_modules([str(plugin_dir)]):
            if module_info.name == "base":
                continue
                
            try:
                module = import_module(f"viewer.env_plugins.{module_info.name}")
                if hasattr(module, 'plugin'):
                    plugin = module.plugin
                    self._plugins[plugin.name] = plugin
                    print(f"Loaded plugin: {plugin.display_name}")
            except Exception as e:
                print(f"Failed to load plugin {module_info.name}: {e}")
    
    def get_plugin(self, name: str) -> Optional[EnvPlugin]:
        return self._plugins.get(name)
    
    def get_all_plugins(self) -> List[EnvPlugin]:
        return list(self._plugins.values())
    
    def identify_environment(self, run_path: Path, summary: Optional[Dict]) -> Optional[str]:
        """Identify which environment handles this run."""
        for plugin in self._plugins.values():
            if plugin.identify(run_path, summary):
                return plugin.name
        return None

# Global registry instance
PLUGIN_REGISTRY = PluginRegistry()
```

### 4. Core Viewer Integration

Replace hardcoded logic with plugin calls:

```python
# Discovery
def discover_evaluation_runs():
    search_paths = []
    for plugin in PLUGIN_REGISTRY.get_all_plugins():
        search_paths.extend(plugin.eval_run_globs)
    
    # Rest of existing logic unchanged
    ...

# Environment identification
def import_evaluation_run_streamlit(run_path: Path, conn):
    # Load summary
    summary = load_summary_if_exists(run_path)
    
    # Identify environment
    env_name = PLUGIN_REGISTRY.identify_environment(run_path, summary)
    if not env_name:
        env_name = 'generic'  # fallback
    
    # Rest of import logic...

# Processing and rendering
def render_trace_visualization(trajectory: Dict, conn):
    env_name = extract_env_name(trajectory)
    plugin = PLUGIN_REGISTRY.get_plugin(env_name)
    
    if plugin:
        processor = plugin.create_processor()
        processed_trace = processor.process_turns(trace_data["trace"]["partition"])
        plugin.render_trace(st, processed_trace)
    else:
        render_generic_trace(trace_data)
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
**Goal**: Create plugin infrastructure and regression tests

#### Tasks
- [ ] Create `env_plugins/base.py` with Protocol definitions
- [ ] Create `PluginRegistry` class with auto-discovery
- [ ] Set up test infrastructure with sample traces
- [ ] Write regression tests for existing Crafter/Sokoban behavior

#### Acceptance Criteria
- [ ] Plugin interface is well-defined and documented
- [ ] Registry can discover and load plugins
- [ ] Tests pass for existing trace processing (baseline)
- [ ] No changes to user-facing behavior yet

#### Files Created
- `src/viewer/env_plugins/__init__.py`
- `src/viewer/env_plugins/base.py`
- `src/viewer/tests/test_plugin_registry.py`
- `src/viewer/tests/test_regression.py`
- `src/viewer/tests/fixtures/` (sample traces)

### Phase 2: Plugin Migration (Week 2)
**Goal**: Migrate Crafter and Sokoban to plugins

#### Tasks
- [ ] Create `env_plugins/crafter.py` with existing logic
- [ ] Create `env_plugins/sokoban.py` with existing logic
- [ ] Update core viewer to use plugins for these environments
- [ ] Run regression tests to ensure no behavioral changes

#### Acceptance Criteria
- [ ] Crafter traces render identically to before
- [ ] Sokoban traces render identically to before
- [ ] All existing functionality works (discovery, import, rendering)
- [ ] Regression tests pass

#### Files Created
- `src/viewer/env_plugins/crafter.py`
- `src/viewer/env_plugins/sokoban.py`

#### Files Modified
- `src/viewer/streamlit_app.py` (integrate registry)

### Phase 3: Complete Migration (Week 3)
**Goal**: Migrate NetHack and MiniGrid, remove hardcoded logic

#### Tasks
- [ ] Create `env_plugins/nethack.py`
- [ ] Create `env_plugins/minigrid.py`
- [ ] Remove hardcoded logic from `streamlit_app.py`
- [ ] Update frontend viewers to use plugin registry
- [ ] Clean up deprecated functions

#### Acceptance Criteria
- [ ] All four environments work through plugin system
- [ ] No hardcoded environment logic in core files
- [ ] File size of `streamlit_app.py` significantly reduced
- [ ] All regression tests pass

#### Files Created
- `src/viewer/env_plugins/nethack.py`
- `src/viewer/env_plugins/minigrid.py`
- `src/viewer/env_plugins/template.py` (for new environments)

#### Files Modified
- `src/viewer/streamlit_app.py` (remove hardcoded logic)
- `src/viewer/frontend/viewers/base.py` (integrate with registry)

### Phase 4: Documentation and Polish (Week 4)
**Goal**: Document new system and create developer guide

#### Tasks
- [ ] Update environment contribution guide
- [ ] Create plugin development documentation
- [ ] Add example of creating new environment plugin
- [ ] Performance testing with multiple environments
- [ ] Error handling and edge case testing

#### Acceptance Criteria
- [ ] Clear documentation for adding new environments
- [ ] Plugin template is easy to follow
- [ ] Error messages are helpful for plugin developers
- [ ] Performance is not degraded

#### Files Created
- `docs/plugin_development.md`
- `docs/adding_new_environment.md`

#### Files Modified
- `Environments/docs/env_contribution_guide.md`
- `README.md` (update with new architecture)

## Testing Strategy

### Regression Tests
Ensure zero behavioral change for existing environments:

```python
# Test data processing equivalence
@pytest.mark.parametrize("trace_file,env_name", [
    ("example_proper_trace.json", "sokoban"),
    ("proper_synth_sdk_traces/crafter_trace.json", "crafter"),
])
def test_processing_equivalence(trace_file, env_name):
    """Ensure new plugin processing matches legacy behavior."""
    trace_data = load_test_trace(trace_file)
    
    # Process with new plugin system
    plugin = PLUGIN_REGISTRY.get_plugin(env_name)
    new_result = plugin.create_processor().process_turns(trace_data["trace"]["partition"])
    
    # Process with legacy function (kept for testing)
    legacy_result = legacy_process_functions[env_name](trace_data)["turns"]
    
    assert len(new_result) == len(legacy_result)
    assert_turns_equivalent(new_result, legacy_result)

# Test rendering smoke tests
def test_rendering_smoke():
    """Ensure rendering functions don't crash."""
    for plugin in PLUGIN_REGISTRY.get_all_plugins():
        mock_st = create_streamlit_mock()
        sample_trace = create_sample_processed_trace()
        
        # Should not raise exception
        plugin.render_trace(mock_st, sample_trace)
```

### Integration Tests
Test the full plugin system:

```python
def test_plugin_discovery():
    """Test that all expected plugins are discovered."""
    registry = PluginRegistry()
    plugin_names = {p.name for p in registry.get_all_plugins()}
    expected = {"crafter", "sokoban", "nethack", "minigrid"}
    assert plugin_names >= expected

def test_environment_identification():
    """Test that runs are correctly identified."""
    registry = PluginRegistry()
    
    # Test explicit identification
    summary = {"environment_name": "crafter"}
    assert registry.identify_environment(Path("/any/path"), summary) == "crafter"
    
    # Test path-based identification
    path = Path("/evals/sokoban/run_123")
    assert registry.identify_environment(path, None) == "sokoban"
```

### Performance Tests
Ensure no performance regression:

```python
def test_plugin_loading_performance():
    """Test that plugin loading doesn't add significant overhead."""
    import time
    
    start = time.time()
    registry = PluginRegistry()
    load_time = time.time() - start
    
    # Should load quickly
    assert load_time < 1.0
    
    # Should have reasonable memory footprint
    assert len(registry.get_all_plugins()) >= 4
```

## Risk Mitigation

### Risk 1: Breaking Existing Functionality
**Mitigation**: Comprehensive regression tests before any code changes

### Risk 2: Performance Impact
**Mitigation**: Lazy loading of plugins, performance benchmarks

### Risk 3: Plugin Development Complexity
**Mitigation**: Clear documentation, template plugin, examples

### Risk 4: Backward Compatibility
**Mitigation**: Maintain existing trace format support, gradual migration

## Success Metrics

### Quantitative
- [ ] Reduce `streamlit_app.py` from 1900+ lines to <1000 lines
- [ ] Zero regression test failures
- [ ] Plugin loading time < 1 second
- [ ] Memory usage increase < 10%

### Qualitative
- [ ] Adding new environment requires only 1 file
- [ ] No manual registry updates needed
- [ ] Clear separation of concerns
- [ ] Maintainable codebase

## Future Enhancements

### Phase 5: Advanced Features (Future)
- [ ] Plugin versioning and compatibility checking
- [ ] Hot-reloading of plugins during development
- [ ] Plugin dependency management
- [ ] Configuration file support for plugins
- [ ] Plugin marketplace/discovery system

### Phase 6: Community Features (Future)
- [ ] External plugin installation via pip
- [ ] Plugin validation and security scanning
- [ ] Community plugin registry
- [ ] Plugin development CLI tools

## Rollback Plan

If critical issues arise:

1. **Immediate**: Revert to previous commit, restore hardcoded logic
2. **Short-term**: Fix specific issues while maintaining plugin architecture
3. **Long-term**: Complete migration with additional testing

## Conclusion

This refactor transforms the trace viewer from a monolithic, hardcoded system into a modular, extensible platform. The plugin architecture enables easy addition of new environments while maintaining the quality and functionality of existing viewers.

The phased approach ensures minimal risk while delivering immediate value through better code organization and long-term value through extensibility.