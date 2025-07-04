let currentTrace = null;
let currentTurnIndex = null;

// Action mapping
const ACTION_NAMES = {
    -1: 'initial state',
    0: 'noop',
    1: 'move_left',
    2: 'move_right',
    3: 'move_up',
    4: 'move_down',
    5: 'do',
    6: 'sleep',
    7: 'place_stone',
    8: 'place_table',
    9: 'place_furnace',
    10: 'place_plant',
    11: 'make_wood_pickaxe',
    12: 'make_stone_pickaxe',
    13: 'make_iron_pickaxe',
    14: 'make_wood_sword',
    15: 'make_stone_sword',
    16: 'make_iron_sword'
};

// Load available traces
async function loadTraceList() {
    try {
        const response = await fetch('/api/traces');
        const traces = await response.json();
        
        const select = document.getElementById('trace-select');
        select.innerHTML = '';
        
        traces.forEach(trace => {
            const option = document.createElement('option');
            option.value = trace.id;
            option.textContent = `${trace.model_name} - ${trace.difficulty} - ${trace.id.substring(0, 8)}`;
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
        const response = await fetch(`/api/trace/${traceId}`);
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
    
    metadataDiv.innerHTML = `
        <div class="metadata-item">
            <span class="metadata-label">Model:</span> ${metadata.model_name}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Difficulty:</span> ${metadata.difficulty}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Seed:</span> ${metadata.seed}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Max Turns:</span> ${metadata.max_turns}
        </div>
    `;
}

// Display timeline
function displayTimeline() {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    
    currentTrace.trace.partition.forEach((partition, index) => {
        const event = partition.events[0];
        const metadata = event.event_metadata;
        
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.innerHTML = `
            <div class="turn-number">Turn ${metadata.turn_number}</div>
            <div class="turn-stats">
                Actions: ${event.environment_compute_steps.length} | 
                Achievements: ${metadata.total_achievements}
                ${metadata.new_achievements.length > 0 ? ' (+' + metadata.new_achievements.length + ')' : ''}
            </div>
        `;
        
        item.addEventListener('click', () => selectTurn(index));
        timeline.appendChild(item);
    });
}

// Display question and reward
function displayQuestionAndReward() {
    const question = currentTrace.dataset.questions[0];
    const reward = currentTrace.dataset.reward_signals[0];
    
    document.getElementById('question-display').innerHTML = `
        <p><strong>Intent:</strong> ${question.intent}</p>
        <p><strong>Criteria:</strong> ${question.criteria}</p>
    `;
    
    document.getElementById('reward-display').innerHTML = `
        <p><strong>Hafner Score:</strong> ${reward.reward.toFixed(2)}%</p>
        <p><strong>Annotation:</strong> ${reward.annotation}</p>
    `;
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

// Display turn details - SIMPLIFIED VERSION
function displayTurnDetails() {
    if (currentTurnIndex === null) return;
    
    const partition = currentTrace.trace.partition[currentTurnIndex];
    const event = partition.events[0];
    const agentStep = event.agent_compute_step;
    const envSteps = event.environment_compute_steps;
    
    let html = '';
    
    // Display actions planned
    if (agentStep.compute_output[0] && agentStep.compute_output[0].outputs) {
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
            // For initial state, show "0. initial state", otherwise show action number
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
    if (event.event_metadata.new_achievements.length > 0) {
        html += '<div class="achievements-section" style="margin-top: 1rem;">';
        html += '<strong>New achievements: </strong>';
        event.event_metadata.new_achievements.forEach(ach => {
            html += `<span class="achievement-badge">${ach}</span>`;
        });
        html += '</div>';
    }
    
    document.getElementById('turn-content').innerHTML = html;
}

// Clear turn details
function clearTurnDetails() {
    document.getElementById('turn-content').innerHTML = '<p class="placeholder">Select a turn from the timeline</p>';
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

// Initial load
loadTraceList();