// static/js/game.js

document.addEventListener('DOMContentLoaded', () => {
    const tableContainer = document.getElementById('table-container');
    const controlsContainer = document.getElementById('controls-container');
    const statusMessage = document.getElementById('status-message');

    let isActionInProgress = false;
    let pnlChartInstance = null; 

    // Function to render or update the P&L chart ---
    const renderOrUpdateChart = (pnlData) => {
        const ctx = document.getElementById('pnlChart').getContext('2d');
        const labels = pnlData.map((_, i) => `Hand ${i}`);

        if (pnlChartInstance) {
            // If chart exists, just update its data
            pnlChartInstance.data.labels = labels;
            pnlChartInstance.data.datasets[0].data = pnlData;
            pnlChartInstance.update();
        } else {
            // If chart doesn't exist, create it
            pnlChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Player Stack',
                        data: pnlData,
                        borderColor: 'rgba(2, 117, 216, 1)',
                        backgroundColor: 'rgba(2, 117, 216, 0.2)',
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: '#f0f0f0' } }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: { color: '#f0f0f0' }
                        },
                        x: {
                            ticks: { color: '#f0f0f0' }
                        }
                    }
                }
            });
        }
    };
    
    const updateUI = (state) => {
        tableContainer.innerHTML = `<img src="data:image/png;base64,${state.table_image}" alt="Poker Table" class="table-image">`;
        controlsContainer.innerHTML = '';
        
        if (state.terminal) {
            statusMessage.textContent = 'Hand Over!';
            const winnerText = state.winners ? `Player ${state.winners.join(', Player ')}` : 'No winner';
            const buttonText = state.win_reason === 'tournament_winner' ? 'Play Again' : 'Next Hand';
            controlsContainer.innerHTML = `
                <div class="game-over">
                    <h2>${state.win_reason.replace('_', ' ').toUpperCase()}</h2>
                    <p><strong>Winner(s):</strong> ${winnerText}</p>
                    <button class="btn btn-new-game" id="new-game-btn">${buttonText}</button>
                </div>
            `;
        } else if (state.to_move === 0) {
            statusMessage.textContent = 'Your Turn';
            let controlsHTML = '<div class="controls">';

            if (state.legal_actions.includes(0)) { // FOLD
                controlsHTML += `<button class="btn btn-fold" data-action="fold">Fold</button>`;
            }

            if (state.legal_actions.includes(1)) { // CALL/CHECK
                const toCall = Math.max(...state.current_bets) - state.current_bets[0];
                const callText = toCall === 0 ? 'Check' : `Call ${toCall}`;
                controlsHTML += `<button class="btn btn-call" data-action="call">${callText}</button>`;
            }

            if (state.legal_actions.includes(2)) { // RAISE/ALL-IN
                if (state.min_raise < state.stacks[0]) {
                    controlsHTML += `
                        <div class="raise-controls">
                            <button class="btn btn-raise" data-action="raise">Raise by</button>
                            <input type="range" class="slider" id="raise-slider" 
                                   min="${state.min_raise}" max="${state.stacks[0]}" value="${state.min_raise}">
                            <output class="raise-output" for="raise-slider">${state.min_raise}</output>
                        </div>
                    `;
                } else {
                    controlsHTML += `<button class="btn btn-raise" data-action="raise" data-amount="${state.stacks[0]}">All-In (${state.stacks[0]})</button>`;
                }
            }
            controlsHTML += '</div>';
            controlsContainer.innerHTML = controlsHTML;
        } else {
            statusMessage.textContent = 'Bot is thinking...';
            setTimeout(triggerBotAction, 1500);
        }

        // --- Call the chart update function ---
        if(state.pnl_history) {
            renderOrUpdateChart(state.pnl_history);
        }
    };

    const performAction = async (action, amount = null) => {
        if (isActionInProgress) return;
        isActionInProgress = true;
        statusMessage.textContent = 'Processing...';

        try {
            const response = await fetch('/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, amount })
            });
            const newState = await response.json();
            if (response.ok) {
                updateUI(newState);
            } else {
                throw new Error(newState.error || 'An unknown error occurred');
            }
        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
        } finally {
            isActionInProgress = false;
        }
    };

    const triggerBotAction = async () => {
        if (isActionInProgress) return;
        isActionInProgress = true;
        
        try {
            const response = await fetch('/bot_action', { method: 'POST' });
            const newState = await response.json();
            if (response.ok) {
                updateUI(newState);
            } else {
                throw new Error(newState.error || 'An unknown error occurred');
            }
        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
        } finally {
            isActionInProgress = false;
        }
    };

    const startNewGame = async () => {
        statusMessage.textContent = 'Starting new hand...';
        controlsContainer.innerHTML = '';
        tableContainer.innerHTML = '<div class="loader"></div>';

        const response = await fetch('/new_game', { method: 'POST' });
        const newState = await response.json();
        updateUI(newState);
    };

    controlsContainer.addEventListener('click', (e) => {
        const target = e.target;
        if (target.tagName !== 'BUTTON') return;

        if (target.id === 'new-game-btn') {
            startNewGame();
            return;
        }

        const action = target.dataset.action;
        if (!action) return;

        let amount = target.dataset.amount || null;
        if (action === 'raise' && !amount) {
            amount = document.getElementById('raise-slider').value;
        }
        performAction(action, amount);
    });

    controlsContainer.addEventListener('input', (e) => {
        if (e.target.id === 'raise-slider') {
            e.target.nextElementSibling.value = e.target.value;
        }
    });
    
    const initializeGame = async () => {
        const response = await fetch('/game_state');
        const initialState = await response.json();
        updateUI(initialState);
    };

    initializeGame();
});

