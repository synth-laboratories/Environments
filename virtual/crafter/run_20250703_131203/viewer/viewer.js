// Auto-generated viewer JavaScript
let currentTrace = null;
let currentTurnIndex = null;

// Environment-specific configuration
const ACTION_NAMES = {"0": "noop", "1": "move_left", "2": "move_right", "3": "move_up", "4": "move_down", "5": "do", "6": "sleep", "7": "place_stone", "8": "place_table", "9": "place_furnace", "10": "place_plant", "11": "make_wood_pickaxe", "12": "make_stone_pickaxe", "13": "make_iron_pickaxe", "14": "make_wood_sword", "15": "make_stone_sword", "16": "make_iron_sword", "-1": "initial state"};

// Load available traces
async function loadTraceList() {
    try {
        // Get eval_dir from window object (injected by server)
        const evalDir = window.EVAL_DIR || '';
        console.log('Loading traces for eval_dir:', evalDir);
        
        const response = await fetch(`/api/traces?eval_dir=${encodeURIComponent(evalDir)}`);
        const traces = await response.json();
        console.log('Loaded traces:', traces);
        
        const select = document.getElementById('trace-select');
        select.innerHTML = '';
        
        traces.forEach(trace => {
            const option = document.createElement('option');
            option.value = trace.id;
            option.textContent = `${trace.model_name} - ${trace.difficulty || 'default'} - ${trace.id.substring(0, 8)}`;
            select.appendChild(option);
        });
        
        if (traces.length > 0) {
            loadTrace(traces[0].id);
        }
    } catch (error) {
        console.error('Failed to load traces:', error);
    }
}

// Load specific trace
async function loadTrace(traceId) {
    try {
        const evalDir = window.EVAL_DIR || '';
        const response = await fetch(`/api/trace/${traceId}?eval_dir=${encodeURIComponent(evalDir)}`);
        const data = await response.json();
        
        currentTrace = data;
        currentTurnIndex = null;
        
        displayTraceInfo();
        displayTimeline();
        displayQuestionAndReward();
        clearTurnDetails();
    } catch (error) {
        console.error('Failed to load trace:', error);
    }
}

// Display trace metadata
function displayTraceInfo() {
    const metadataDiv = document.getElementById('trace-metadata');
    const metadata = currentTrace.trace.metadata;
    
    let html = '';
    for (const [key, value] of Object.entries(metadata)) {
        html += `
            <div class="metadata-item">
                <span class="metadata-label">${formatLabel(key)}:</span> ${value}
            </div>
        `;
    }
    
    metadataDiv.innerHTML = html;
}

// Display timeline
function displayTimeline() {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    
    currentTrace.trace.partition.forEach((partition, index) => {
        const event = partition.events[0];
        const metadata = event.event_metadata || {};
        
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.innerHTML = `
            <div class="turn-number">Turn ${metadata.turn_number || index + 1}</div>
            <div class="turn-stats">${getTimelineStats(event)}</div>
        `;
        
        item.addEventListener('click', () => selectTurn(index));
        timeline.appendChild(item);
    });
}

// Display question and reward
function displayQuestionAndReward() {
    const dataset = currentTrace.dataset;
    
    if (dataset.questions && dataset.questions.length > 0) {
        const question = dataset.questions[0];
        document.getElementById('question-display').innerHTML = `
            <p><strong>Intent:</strong> ${question.intent || 'N/A'}</p>
            <p><strong>Criteria:</strong> ${question.criteria || 'N/A'}</p>
        `;
    } else {
        document.getElementById('question-display').innerHTML = '<p>No training question available</p>';
    }
    
    if (dataset.reward_signals && dataset.reward_signals.length > 0) {
        const reward = dataset.reward_signals[0];
        document.getElementById('reward-display').innerHTML = `
            <p><strong>Score:</strong> ${(reward.reward || 0).toFixed(2)}%</p>
            <p><strong>Annotation:</strong> ${reward.annotation || 'N/A'}</p>
        `;
    } else {
        document.getElementById('reward-display').innerHTML = '<p>No reward signal available</p>';
    }
}

// Select turn
function selectTurn(index) {
    currentTurnIndex = index;
    
    // Update timeline selection
    document.querySelectorAll('.timeline-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });
    
    // Display turn details
    displayTurnDetails();
}

// Clear turn details
function clearTurnDetails() {
    document.getElementById('turn-content').innerHTML = '<p class="placeholder">Select a turn from the timeline</p>';
}

// Helper functions
function formatLabel(key) {
    return key.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

function getTimelineStats(event) {
    // Default implementation - can be overridden
    const envSteps = event.environment_compute_steps || [];
    return `Actions: ${envSteps.length}`;
}

// Event listeners
document.getElementById('trace-select').addEventListener('change', (e) => {
    if (e.target.value) {
        loadTrace(e.target.value);
    }
});

document.getElementById('refresh-btn').addEventListener('click', () => {
    loadTraceList();
});

// Environment-specific display function

// Display turn details - SIMPLIFIED VERSION FOR CRAFTER
function displayTurnDetails() {
    if (currentTurnIndex === null) return;
    
    const partition = currentTrace.trace.partition[currentTurnIndex];
    const event = partition.events[0];
    const agentStep = event.agent_compute_step;
    const envSteps = event.environment_compute_steps;
    
    let html = '';
    
    // Display actions planned
    if (agentStep && agentStep.compute_output && agentStep.compute_output[0] && agentStep.compute_output[0].outputs) {
        const outputs = agentStep.compute_output[0].outputs;
        const actionNames = outputs.actions.map(idx => `${ACTION_NAMES[idx] || 'unknown'}`);
        html += `
            <div class="actions-box" style="margin-bottom: 1.5rem;">
                <strong>Turn ${event.event_metadata.turn_number} Actions:</strong> ${actionNames.join(' → ')}
            </div>
        `;
    }
    
    // Display all images in a row
    html += '<div class="images-row">';
    envSteps.forEach((step, i) => {
        const outputs = step.compute_output[0].outputs;
        const actionName = ACTION_NAMES[outputs.action_index] || 'unknown';
        
        if (outputs.image_base64) {
            const stepNumber = outputs.action_index === -1 ? 0 : i;
            html += `
                <div class="image-container">
                    <img src="data:image/png;base64,${outputs.image_base64}" class="env-image" alt="Game state">
                    <div class="image-caption">${stepNumber}. ${actionName}</div>
                </div>
            `;
        }
    });
    html += '</div>';
    
    // New achievements
    if (event.event_metadata.new_achievements && event.event_metadata.new_achievements.length > 0) {
        html += '<div class="achievements-section" style="margin-top: 1rem;">';
        html += '<strong>New achievements: </strong>';
        event.event_metadata.new_achievements.forEach(ach => {
            html += `<span class="achievement-badge">${ach}</span>`;
        });
        html += '</div>';
    }
    
    // Add debug section with messages and tool calls
    html += `
        <div class="debug-section">
            <div class="debug-toggle" onclick="toggleDebugInfo()">
                🔍 View Messages & Tool Calls
            </div>
            <div id="debug-content" class="debug-content">
                ${getDebugContent(agentStep, envSteps)}
            </div>
        </div>
    `;
    
    document.getElementById('turn-content').innerHTML = html;
}

// Toggle debug information visibility
function toggleDebugInfo() {
    const debugContent = document.getElementById('debug-content');
    debugContent.classList.toggle('show');
}

// Get debug content (messages and tool outputs)
function getDebugContent(agentStep, envSteps) {
    let html = '<div class="message-section">';
    
    // Display agent messages
    if (agentStep && agentStep.compute_input && agentStep.compute_input[0] && agentStep.compute_input[0].messages) {
        html += '<h4>Messages:</h4>';
        agentStep.compute_input[0].messages.forEach(msg => {
            html += `
                <div class="message">
                    <div class="message-role">${msg.role.toUpperCase()}</div>
                    <div class="message-content">${escapeHtml(msg.content)}</div>
                </div>
            `;
        });
    }
    
    // Display agent output (reconstructed tool call from actions)
    if (agentStep && agentStep.compute_output && agentStep.compute_output[0]) {
        const output = agentStep.compute_output[0];
        
        // For Crafter, the tool call would have been crafter_interact with the actions list
        if (output.outputs && output.outputs.actions) {
            html += '<h4>Tool Call (reconstructed):</h4>';
            const actionNames = output.outputs.actions.map(idx => ACTION_NAMES[idx] || `action_${idx}`);
            const toolCall = {
                tool: 'crafter_interact',
                arguments: {
                    actions_list: actionNames,
                    reasoning: '(reasoning not captured in trace)'
                }
            };
            html += `
                <div class="tool-output">
                    <strong>Tool:</strong> ${toolCall.tool}<br>
                    <strong>Arguments:</strong><br>
                    <pre>${JSON.stringify(toolCall.arguments, null, 2)}</pre>
                </div>
            `;
        }
    }
    
    // Display environment outputs in a cleaner format
    if (envSteps && envSteps.length > 0) {
        html += '<h4>Tool Results:</h4>';
        html += '<div class="tool-output">';
        html += '<strong>Actions executed:</strong><br>';
        envSteps.forEach((step, idx) => {
            if (step.compute_output && step.compute_output[0] && step.compute_output[0].outputs) {
                const outputs = step.compute_output[0].outputs;
                const actionName = ACTION_NAMES[outputs.action_index] || outputs.action_index;
                const reward = outputs.reward;
                html += `${idx + 1}. ${actionName} (reward: ${reward.toFixed(3)})<br>`;
            }
        });
        
        // Add summary of final state
        const lastStep = envSteps[envSteps.length - 1];
        if (lastStep && lastStep.compute_output && lastStep.compute_output[0]) {
            const outputs = lastStep.compute_output[0].outputs;
            html += `<br><strong>Final state:</strong> `;
            html += `Total reward: ${outputs.total_reward.toFixed(3)}, `;
            html += `Steps: ${outputs.num_steps}, `;
            html += `Status: ${outputs.terminated ? 'TERMINATED' : outputs.truncated ? 'TRUNCATED' : 'ACTIVE'}`;
        }
        html += '</div>';
    }
    
    html += '</div>';
    return html;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Override timeline stats for Crafter
function getTimelineStats(event) {
    const metadata = event.event_metadata || {};
    const envSteps = event.environment_compute_steps || [];
    const newAchievements = metadata.new_achievements || [];
    
    let stats = `Actions: ${envSteps.length}`;
    if (metadata.total_achievements !== undefined) {
        stats += ` | Achievements: ${metadata.total_achievements}`;
        if (newAchievements.length > 0) {
            stats += ` (+${newAchievements.length})`;
        }
    }
    return stats;
}


// Initial load
loadTraceList();
