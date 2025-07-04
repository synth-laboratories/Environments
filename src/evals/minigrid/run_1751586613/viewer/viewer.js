// Auto-generated viewer JavaScript
let currentTrace = null;
let currentTurnIndex = null;

// Environment-specific configuration
const ACTION_NAMES = {"0": "turn_left", "1": "turn_right", "2": "forward", "3": "pickup", "4": "drop", "5": "toggle", "6": "done"};

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

// MiniGrid-specific viewer JavaScript

// Display turn details for MiniGrid
function displayTurnDetails() {
    if (currentTurnIndex === null) return;
    
    const partition = currentTrace.trace.partition[currentTurnIndex];
    const event = partition.events[0];
    const agentStep = event.agent_compute_step;
    const envSteps = event.environment_compute_steps;
    
    let html = '';
    
    // Display turn number and event type
    html += `
        <div class="actions-box" style="margin-bottom: 1.5rem;">
            <strong>Turn ${currentTurnIndex + 1}</strong> - Event: ${event.event_type || 'unknown'}
        </div>
    `;
    
    // Display grid visualization (simplified for now)
    html += '<div class="grid-container">';
    html += generateGridVisualization([2, 2], 0); // Default position and direction
    html += '</div>';
    
    // Show event timing
    if (event.opened && event.closed) {
        const duration = ((event.closed - event.opened) * 1000).toFixed(1);
        html += `
            <div class="timing-info" style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                Duration: ${duration}ms
            </div>
        `;
    }
    
    document.getElementById('turn-content').innerHTML = html;
}

// Generate grid visualization
function generateGridVisualization(agentPos, agentDir) {
    const gridSize = 5; // Default for MiniGrid-Empty-5x5-v0
    const cellSize = 40;
    
    let html = `<div class="minigrid" style="display: inline-block; border: 2px solid #333;">`;
    
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            let cellClass = 'cell';
            let cellContent = '';
            
            // Check if this is the agent position
            if (agentPos[0] === x && agentPos[1] === y) {
                cellClass += ' agent';
                // Add direction indicator
                const directions = ['→', '↓', '←', '↑'];
                cellContent = directions[agentDir] || '?';
            }
            
            // Check if this is the goal (bottom-right corner)
            if (x === gridSize - 1 && y === gridSize - 1) {
                cellClass += ' goal';
                if (!cellContent) cellContent = '🎯';
            }
            
            // Check if this is a wall (border)
            if (x === 0 || x === gridSize - 1 || y === 0 || y === gridSize - 1) {
                if (!(x === gridSize - 1 && y === gridSize - 1)) { // Not the goal
                    cellClass += ' wall';
                }
            }
            
            html += `<div class="${cellClass}" style="width: ${cellSize}px; height: ${cellSize}px; display: inline-block; border: 1px solid #ccc; text-align: center; line-height: ${cellSize}px; vertical-align: top;">${cellContent}</div>`;
        }
        html += '<br>';
    }
    
    html += '</div>';
    return html;
}

// Override timeline stats for MiniGrid
function getTimelineStats(event) {
    const eventType = event.event_type || 'unknown';
    let stats = `Type: ${eventType}`;
    
    if (event.opened && event.closed) {
        const duration = ((event.closed - event.opened) * 1000).toFixed(0);
        stats += ` | ${duration}ms`;
    }
    
    return stats;
}


// Initial load
loadTraceList();
