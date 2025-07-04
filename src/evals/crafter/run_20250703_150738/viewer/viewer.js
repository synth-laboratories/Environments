let currentTrace = null;
let currentTurnIndex = null;

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

// Display turn details
function displayTurnDetails() {
    if (currentTurnIndex === null) return;
    
    const partition = currentTrace.trace.partition[currentTurnIndex];
    const event = partition.events[0];
    const agentStep = event.agent_compute_step;
    const envSteps = event.environment_compute_steps;
    
    let html = '<div class="agent-section">';
    html += '<h3>Agent Decision</h3>';
    
    // Display messages
    if (agentStep.compute_input[0] && agentStep.compute_input[0].messages) {
        agentStep.compute_input[0].messages.forEach(msg => {
            html += `
                <div class="message-box">
                    <div class="role">${msg.role}</div>
                    <div class="content">${msg.content}</div>
                </div>
            `;
        });
    }
    
    // Display actions
    if (agentStep.compute_output[0] && agentStep.compute_output[0].outputs) {
        const outputs = agentStep.compute_output[0].outputs;
        html += `
            <div class="actions-box">
                <strong>Actions:</strong> ${outputs.actions.join(', ')} (${outputs.action_count} total)
            </div>
        `;
    }
    
    html += '</div>';
    
    // Environment steps
    html += '<div class="environment-section">';
    html += '<h3>Environment Response</h3>';
    
    envSteps.forEach((step, i) => {
        const outputs = step.compute_output[0].outputs;
        html += `<div class="env-step">`;
        html += `<h4>Action ${i + 1}: ${outputs.action_index}</h4>`;
        
        // Stats
        if (outputs.player_stats) {
            html += `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Health</div>
                        <div class="stat-value">${outputs.player_stats.health}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Food</div>
                        <div class="stat-value">${outputs.player_stats.food}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Drink</div>
                        <div class="stat-value">${outputs.player_stats.drink}</div>
                    </div>
                </div>
            `;
        }
        
        html += `<p>Reward: ${outputs.reward} (Total: ${outputs.total_reward.toFixed(3)})</p>`;
        html += `<p>Steps: ${outputs.num_steps}</p>`;
        
        // Image
        if (outputs.image_base64) {
            html += `<img src="data:image/png;base64,${outputs.image_base64}" class="env-image" alt="Game state">`;
        }
        
        // Termination status
        if (outputs.terminated || outputs.truncated) {
            html += `<p style="color: red; font-weight: bold;">
                ${outputs.terminated ? 'TERMINATED' : 'TRUNCATED'}
            </p>`;
        }
        
        html += '</div>';
    });
    
    html += '</div>';
    
    // New achievements
    if (event.event_metadata.new_achievements.length > 0) {
        html += '<div class="achievements-section">';
        html += '<h3>New Achievements Unlocked</h3>';
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