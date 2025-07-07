# Known Issues

## Model Compatibility Issues

### Gemini Tool Calling Failure
**Environment**: Crafter Classic Multi-Action Evaluation  
**Model**: `gemini-2.0-flash-exp`  
**Issue**: Model fails to return tool calls when using BaseTool-based tools  
**Error**: `AssertionError: Response object didn't have tool call`  
**Status**: Open  
**Date**: January 2025  

**Description**: 
The Gemini 2.0 Flash model does not consistently return tool calls when using the synth-ai BaseTool framework in the Crafter evaluation environment. While Claude 3.5 Haiku and GPT-4o-mini work correctly with the same tool definitions, Gemini fails to generate the expected tool call responses.

**Impact**: 
- Gemini models cannot be evaluated in multi-action Crafter environments
- Limits model comparison capabilities
- May affect other environments using similar tool patterns

**Workaround**: 
- Currently skipping Gemini models in evaluations
- Focus comparisons on Claude and OpenAI models

**Next Steps**: 
- Investigate Gemini-specific tool formatting requirements
- Check if different tool schema formats work better with Gemini
- Consider model-specific tool adapters in synth-ai framework

## Environment-Specific Issues

### Max Turns Display Bug
**Environment**: Crafter Classic  
**Issue**: Progress display shows incorrect max_turns value (15 instead of configured 30)  
**Status**: Minor - cosmetic issue only  
**Impact**: No functional impact, configuration is applied correctly  

**Description**:
The turn progress display shows "Turn X/15" instead of "Turn X/30" even when max_turns is configured to 30 in eval_config.toml. The actual evaluation runs for the correct number of turns, this is purely a display issue. 



MINIGRID environment does not have proper rending / validation for e.g. yellow doors and such