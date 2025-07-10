// Cricket Century Prediction Platform JavaScript

// Sample data for the platform
const playersData = [
    {
        id: "virat_kohli",
        name: "Virat Kohli",
        country: "India",
        role: "Batsman",
        matches: 254,
        runs: 12169,
        average: 57.32,
        centuries: 43,
        strikeRate: 93.17
    },
    {
        id: "steve_smith",
        name: "Steve Smith",
        country: "Australia",
        role: "Batsman",
        matches: 129,
        runs: 7540,
        average: 62.84,
        centuries: 27,
        strikeRate: 91.63
    },
    {
        id: "david_warner",
        name: "David Warner",
        country: "Australia",
        role: "Batsman",
        matches: 128,
        runs: 5455,
        average: 45.3,
        centuries: 18,
        strikeRate: 95.4
    },
    {
        id: "kane_williamson",
        name: "Kane Williamson",
        country: "New Zealand",
        role: "Batsman",
        matches: 83,
        runs: 6173,
        average: 54.31,
        centuries: 24,
        strikeRate: 51.25
    },
    {
        id: "joe_root",
        name: "Joe Root",
        country: "England",
        role: "Batsman",
        matches: 118,
        runs: 9460,
        average: 50.85,
        centuries: 24,
        strikeRate: 54.75
    },
    {
        id: "pat_cummins",
        name: "Pat Cummins",
        country: "Australia",
        role: "Bowler",
        matches: 44,
        runs: 766,
        average: 19.64,
        centuries: 0,
        strikeRate: 45.23
    }
];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    initializePlayerStats();
    populatePlayerSuggestions();
});

// Navigation functions
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Remove active class from all nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected page
    document.getElementById(pageId).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Initialize page-specific content
    switch(pageId) {
        case 'dashboard':
            initializeDashboard();
            break;
        case 'player-analysis':
            initializePlayerAnalysis();
            break;
        case 'model-comparison':
            initializeModelComparison();
            break;
        case 'data-explorer':
            initializeDataExplorer();
            break;
    }
}

// Dashboard initialization
function initializeDashboard() {
    createPerformanceChart();
    createFormatChart();
}

function createPerformanceChart() {
    const dates = [];
    const tensorflow = [];
    const pytorch = [];
    const ensemble = [];
    
    for (let i = 0; i < 30; i++) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        dates.unshift(date.toISOString().split('T')[0]);
        
        tensorflow.unshift(0.80 + Math.random() * 0.1);
        pytorch.unshift(0.78 + Math.random() * 0.1);
        ensemble.unshift(0.82 + Math.random() * 0.08);
    }
    
    const trace1 = {
        x: dates,
        y: tensorflow,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'TensorFlow',
        line: {color: '#3498db'}
    };
    
    const trace2 = {
        x: dates,
        y: pytorch,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'PyTorch',
        line: {color: '#e74c3c'}
    };
    
    const trace3 = {
        x: dates,
        y: ensemble,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Ensemble',
        line: {color: '#27ae60'}
    };
    
    const layout = {
        title: '',
        xaxis: {title: 'Date'},
        yaxis: {title: 'Accuracy'},
        margin: {t: 0}
    };
    
    Plotly.newPlot('performance-chart', [trace1, trace2, trace3], layout);
}

function createFormatChart() {
    const trace = {
        x: ['Test', 'ODI', 'T20', 'T10'],
        y: [145, 289, 167, 43],
        type: 'bar',
        name: 'Actual Centuries',
        marker: {color: '#3498db'}
    };
    
    const trace2 = {
        x: ['Test', 'ODI', 'T20', 'T10'],
        y: [156, 301, 178, 39],
        type: 'bar',
        name: 'Predicted Centuries',
        marker: {color: '#e74c3c'}
    };
    
    const layout = {
        title: '',
        xaxis: {title: 'Format'},
        yaxis: {title: 'Centuries'},
        barmode: 'group',
        margin: {t: 0}
    };
    
    Plotly.newPlot('format-chart', [trace, trace2], layout);
}

// Prediction functions
function makePrediction() {
    const playerName = document.getElementById('player-name').value;
    const format = document.getElementById('match-format').value;
    const opposition = document.getElementById('opposition').value;
    const venue = document.getElementById('venue').value;
    const weather = document.getElementById('weather').value;
    const pitchType = document.getElementById('pitch-type').value;
    const temperature = document.getElementById('temperature').value;
    
    // Simple prediction algorithm based on player and conditions
    let basePrediction = 0.5;
    
    // Player factor
    const player = playersData.find(p => p.name.toLowerCase() === playerName.toLowerCase());
    if (player) {
        basePrediction += (player.average - 40) / 100;
        basePrediction += player.centuries / 200;
    }
    
    // Format factor
    const formatFactors = {'Test': 0.1, 'ODI': 0.05, 'T20': -0.1, 'T10': -0.2};
    basePrediction += formatFactors[format] || 0;
    
    // Weather factor
    const weatherFactors = {'Clear': 0.1, 'Overcast': 0.05, 'Light Rain': -0.05, 'Heavy Rain': -0.2};
    basePrediction += weatherFactors[weather] || 0;
    
    // Pitch factor
    const pitchFactors = {'Flat': 0.15, 'Green': -0.1, 'Dusty': 0.05, 'Cracked': -0.05};
    basePrediction += pitchFactors[pitchType] || 0;
    
    // Temperature factor
    const temp = parseInt(temperature);
    if (temp > 30) basePrediction += 0.05;
    if (temp < 15) basePrediction -= 0.1;
    
    // Ensure prediction is between 0 and 1
    basePrediction = Math.max(0.1, Math.min(0.95, basePrediction));
    
    const predictionPercentage = Math.round(basePrediction * 100);
    
    // Update UI
    document.getElementById('prediction-value').textContent = predictionPercentage + '%';
    document.getElementById('confidence-fill').style.width = predictionPercentage + '%';
    
    let confidenceLabel = 'Low Confidence';
    if (predictionPercentage > 70) confidenceLabel = 'High Confidence';
    else if (predictionPercentage > 50) confidenceLabel = 'Medium Confidence';
    
    document.getElementById('confidence-label').textContent = confidenceLabel;
    
    // Generate factors
    const factors = generateFactors(player, format, weather, pitchType, temperature);
    displayFactors(factors);
    
    // Show results
    document.getElementById('prediction-results').style.display = 'block';
}

function generateFactors(player, format, weather, pitchType, temperature) {
    const factors = [];
    
    if (player) {
        if (player.average > 50) {
            factors.push({name: 'High batting average', impact: 'positive'});
        }
        if (player.centuries > 20) {
            factors.push({name: 'Proven century scorer', impact: 'positive'});
        }
        if (player.strikeRate > 90) {
            factors.push({name: 'Aggressive batting style', impact: format === 'T20' || format === 'T10' ? 'positive' : 'neutral'});
        }
    }
    
    if (format === 'Test') {
        factors.push({name: 'Test format favors centuries', impact: 'positive'});
    } else if (format === 'T10') {
        factors.push({name: 'T10 format limits century chances', impact: 'negative'});
    }
    
    if (weather === 'Clear') {
        factors.push({name: 'Clear weather conditions', impact: 'positive'});
    } else if (weather === 'Heavy Rain') {
        factors.push({name: 'Heavy rain expected', impact: 'negative'});
    }
    
    if (pitchType === 'Flat') {
        factors.push({name: 'Batting-friendly pitch', impact: 'positive'});
    } else if (pitchType === 'Green') {
        factors.push({name: 'Pace-friendly conditions', impact: 'negative'});
    }
    
    const temp = parseInt(document.getElementById('temperature').value);
    if (temp > 30) {
        factors.push({name: 'Hot weather favors batting', impact: 'positive'});
    } else if (temp < 15) {
        factors.push({name: 'Cold conditions', impact: 'negative'});
    }
    
    return factors;
}

function displayFactors(factors) {
    const factorsList = document.getElementById('factors-list');
    factorsList.innerHTML = '';
    
    factors.forEach(factor => {
        const factorItem = document.createElement('div');
        factorItem.className = 'factor-item';
        
        const factorName = document.createElement('span');
        factorName.className = 'factor-name';
        factorName.textContent = factor.name;
        
        const factorImpact = document.createElement('span');
        factorImpact.className = `factor-impact factor-${factor.impact}`;
        factorImpact.textContent = factor.impact.charAt(0).toUpperCase() + factor.impact.slice(1);
        
        factorItem.appendChild(factorName);
        factorItem.appendChild(factorImpact);
        factorsList.appendChild(factorItem);
    });
}

// Player search and analysis functions
function searchPlayers() {
    const query = document.getElementById('player-search').value.toLowerCase();
    const suggestions = document.getElementById('player-suggestions');
    
    if (query.length < 2) {
        suggestions.style.display = 'none';
        return;
    }
    
    const matchedPlayers = playersData.filter(player => 
        player.name.toLowerCase().includes(query)
    );
    
    suggestions.innerHTML = '';
    
    matchedPlayers.forEach(player => {
        const item = document.createElement('div');
        item.className = 'suggestion-item';
        item.textContent = `${player.name} (${player.country})`;
        item.onclick = () => selectPlayer(player);
        suggestions.appendChild(item);
    });
    
    suggestions.style.display = matchedPlayers.length > 0 ? 'block' : 'none';
}

function selectPlayer(player) {
    document.getElementById('player-search').value = player.name;
    document.getElementById('player-suggestions').style.display = 'none';
    displayPlayerProfile(player);
    createPlayerCharts(player);
}

function populatePlayerSuggestions() {
    // Initialize with Steve Smith
    const steveSmith = playersData.find(p => p.name === 'Steve Smith');
    if (steveSmith) {
        displayPlayerProfile(steveSmith);
        createPlayerCharts(steveSmith);
    }
}

function displayPlayerProfile(player) {
    const profileDiv = document.getElementById('player-profile');
    profileDiv.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div>
                <h3>${player.name}</h3>
                <p><strong>Country:</strong> ${player.country}</p>
                <p><strong>Role:</strong> ${player.role}</p>
            </div>
            <div>
                <p><strong>Matches:</strong> ${player.matches}</p>
                <p><strong>Runs:</strong> ${player.runs.toLocaleString()}</p>
                <p><strong>Average:</strong> ${player.average}</p>
            </div>
            <div>
                <p><strong>Centuries:</strong> ${player.centuries}</p>
                <p><strong>Strike Rate:</strong> ${player.strikeRate}</p>
            </div>
        </div>
    `;
}

function createPlayerCharts(player) {
    // Form trend chart
    const formData = [];
    for (let i = 0; i < 10; i++) {
        formData.push(Math.floor(Math.random() * 150) + 10);
    }
    
    const formTrace = {
        x: Array.from({length: 10}, (_, i) => `Match ${10-i}`),
        y: formData,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Runs Scored',
        line: {color: '#3498db'}
    };
    
    Plotly.newPlot('form-chart', [formTrace], {
        title: '',
        xaxis: {title: 'Recent Matches'},
        yaxis: {title: 'Runs'},
        margin: {t: 0}
    });
    
    // Venue performance chart
    const venues = ['MCG', 'SCG', 'Lord\'s', 'Eden Gardens', 'Oval'];
    const venueScores = venues.map(() => Math.floor(Math.random() * 80) + 20);
    
    const venueTrace = {
        x: venues,
        y: venueScores,
        type: 'bar',
        marker: {color: '#27ae60'}
    };
    
    Plotly.newPlot('venue-chart', [venueTrace], {
        title: '',
        xaxis: {title: 'Venues'},
        yaxis: {title: 'Average Score'},
        margin: {t: 0}
    });
}

function initializePlayerAnalysis() {
    populatePlayerSuggestions();
}

// Model comparison functions
function initializeModelComparison() {
    createModelComparisonChart();
}

function createModelComparisonChart() {
    const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
    const tensorflow = [85.7, 82.3, 87.1, 84.6];
    const pytorch = [83.4, 80.1, 85.9, 82.9];
    const ensemble = [88.2, 86.7, 89.3, 88.0];
    
    const trace1 = {
        x: metrics,
        y: tensorflow,
        type: 'bar',
        name: 'TensorFlow',
        marker: {color: '#3498db'}
    };
    
    const trace2 = {
        x: metrics,
        y: pytorch,
        type: 'bar',
        name: 'PyTorch',
        marker: {color: '#e74c3c'}
    };
    
    const trace3 = {
        x: metrics,
        y: ensemble,
        type: 'bar',
        name: 'Ensemble',
        marker: {color: '#27ae60'}
    };
    
    const layout = {
        title: '',
        xaxis: {title: 'Metrics'},
        yaxis: {title: 'Score (%)'},
        barmode: 'group',
        margin: {t: 0}
    };
    
    Plotly.newPlot('model-comparison-chart', [trace1, trace2, trace3], layout);
}

// Data explorer functions
function initializeDataExplorer() {
    initializePlayerStats();
    createCountryDistributionChart();
    createFormatScoresChart();
}

function initializePlayerStats() {
    const tableBody = document.getElementById('stats-table-body');
    tableBody.innerHTML = '';
    
    playersData.forEach(player => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${player.name}</td>
            <td>${player.country}</td>
            <td>${player.matches}</td>
            <td>${player.runs.toLocaleString()}</td>
            <td>${player.average}</td>
            <td>${player.centuries}</td>
            <td>${player.strikeRate}</td>
        `;
        tableBody.appendChild(row);
    });
}

function createCountryDistributionChart() {
    const countries = {};
    playersData.forEach(player => {
        countries[player.country] = (countries[player.country] || 0) + player.centuries;
    });
    
    const trace = {
        labels: Object.keys(countries),
        values: Object.values(countries),
        type: 'pie',
        textinfo: 'label+percent',
        textposition: 'outside'
    };
    
    Plotly.newPlot('country-distribution-chart', [trace], {
        title: '',
        margin: {t: 0}
    });
}

function createFormatScoresChart() {
    const formats = ['Test', 'ODI', 'T20', 'T10'];
    const avgScores = [45, 35, 25, 15];
    
    const trace = {
        x: formats,
        y: avgScores,
        type: 'bar',
        marker: {
            color: ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
        }
    };
    
    Plotly.newPlot('format-scores-chart', [trace], {
        title: '',
        xaxis: {title: 'Format'},
        yaxis: {title: 'Average Score'},
        margin: {t: 0}
    });
}

function applyFilters() {
    const formatFilter = document.getElementById('format-filter').value;
    const countryFilter = document.getElementById('country-filter').value;
    
    let filteredData = playersData;
    
    if (countryFilter !== 'all') {
        filteredData = filteredData.filter(player => player.country === countryFilter);
    }
    
    // Update table with filtered data
    const tableBody = document.getElementById('stats-table-body');
    tableBody.innerHTML = '';
    
    filteredData.forEach(player => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${player.name}</td>
            <td>${player.country}</td>
            <td>${player.matches}</td>
            <td>${player.runs.toLocaleString()}</td>
            <td>${player.average}</td>
            <td>${player.centuries}</td>
            <td>${player.strikeRate}</td>
        `;
        tableBody.appendChild(row);
    });
    
    // Update charts based on filtered data
    updateChartsWithFilter(filteredData);
}

function updateChartsWithFilter(filteredData) {
    // Update country distribution chart
    const countries = {};
    filteredData.forEach(player => {
        countries[player.country] = (countries[player.country] || 0) + player.centuries;
    });
    
    const trace = {
        labels: Object.keys(countries),
        values: Object.values(countries),
        type: 'pie',
        textinfo: 'label+percent',
        textposition: 'outside'
    };
    
    Plotly.newPlot('country-distribution-chart', [trace], {
        title: '',
        margin: {t: 0}
    });
}

// Utility functions
function formatNumber(num) {
    return num.toLocaleString();
}

function getRandomColor() {
    const colors = ['#3498db', '#27ae60', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c'];
    return colors[Math.floor(Math.random() * colors.length)];
}

// Hide suggestions when clicking outside
document.addEventListener('click', function(event) {
    const searchBox = document.getElementById('player-search');
    const suggestions = document.getElementById('player-suggestions');
    
    if (!searchBox.contains(event.target) && !suggestions.contains(event.target)) {
        suggestions.style.display = 'none';
    }
});