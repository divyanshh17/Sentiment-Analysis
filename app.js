// Main application JavaScript

let currentResults = null;
let currentText = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const submitFeedbackBtn = document.getElementById('submitFeedback');
    const exportBtn = document.getElementById('exportBtn');
    const depthSlider = document.getElementById('depthSlider');
    
    analyzeBtn.addEventListener('click', analyzeText);
    submitFeedbackBtn.addEventListener('click', submitFeedback);
    exportBtn.addEventListener('click', exportResults);
    depthSlider.addEventListener('input', updateHighlightDepth);
    
    // Feedback buttons
    document.querySelectorAll('.btn-feedback').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-feedback').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
    
    // Allow Enter+Ctrl to analyze
    document.getElementById('textInput').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeText();
        }
    });
});

async function analyzeText() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    currentText = text;
    
    const method = document.getElementById('methodSelect').value;
    const includeCounterfactuals = document.getElementById('includeCounterfactuals').checked;
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Disable button and show loading
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Store original HTML if first time
    if (!resultsSection.dataset.originalHTML) {
        resultsSection.dataset.originalHTML = resultsSection.innerHTML;
    }
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.textContent = 'Analyzing text...';
    resultsSection.innerHTML = '';
    resultsSection.appendChild(loadingDiv);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                explanation_method: method,
                include_counterfactuals: includeCounterfactuals
            })
        });
        
        // Parse response
        let data;
        try {
            data = await response.json();
        } catch (parseError) {
            throw new Error('Invalid response from server. Please try again.');
        }
        
        // Check for error in response
        if (!response.ok) {
            const errorMsg = data.error || data.details || `Server error (${response.status})`;
            throw new Error(errorMsg);
        }
        
        // Check if response has error field
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Validate response structure
        if (!data.label || data.confidence === undefined) {
            throw new Error('Invalid response format from server.');
        }
        
        currentResults = data;
        
        // Restore original HTML structure
        if (resultsSection.dataset.originalHTML) {
            resultsSection.innerHTML = resultsSection.dataset.originalHTML;
        }
        
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Unable to analyze text. Please try again later.');
    } finally {
        // Re-enable button
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
    }
}

function showError(message) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.innerHTML = `
        <strong>Error:</strong> ${message}
        <br><br>
        <button onclick="document.getElementById('resultsSection').style.display='none'" 
                style="padding: 8px 16px; background: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer;">
            Dismiss
        </button>
    `;
    resultsSection.innerHTML = '';
    resultsSection.appendChild(errorDiv);
}

function displayResults(data) {
    // Restore results section HTML structure if it was replaced
    const resultsSection = document.getElementById('resultsSection');
    if (!resultsSection.querySelector('.prediction-card')) {
        // Reload page to restore structure - in production, rebuild HTML dynamically
        window.location.reload();
        return;
    }
    
    updatePrediction(data);
    updateRationale(data);
    updateAttributions(data);
    updateCounterfactuals(data);
}

function updatePrediction(data) {
    const labelBadge = document.getElementById('labelBadge');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const qualityBadge = document.getElementById('qualityBadge');
    
    // Update label
    labelBadge.textContent = data.label;
    labelBadge.className = `label-badge ${data.label}`;
    
    // Update confidence
    const confidencePercent = (data.confidence * 100).toFixed(1);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;
    
    // Update quality
    qualityBadge.textContent = `Quality: ${data.explainability_quality}`;
    qualityBadge.className = `quality-badge ${data.explainability_quality}`;
}

function updateRationale(data) {
    const rationaleText = document.getElementById('rationaleText');
    rationaleText.textContent = data.rationale;
}

function updateAttributions(data) {
    const textWithHighlights = document.getElementById('textWithHighlights');
    const attributions = data.attributions;
    
    if (!attributions || attributions.length === 0) {
        textWithHighlights.textContent = currentText;
        return;
    }
    
    // Create a map of token to attribution score
    const tokenScores = new Map();
    attributions.forEach(attr => {
        const token = attr.token.replace('Ġ', '').replace('▁', '');
        tokenScores.set(token.toLowerCase(), attr.score);
    });
    
    // Highlight text
    let highlightedHTML = currentText;
    
    // Sort attributions by absolute score
    const sortedAttributions = [...attributions].sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
    
    // Create spans for each token
    sortedAttributions.forEach(attr => {
        const token = attr.token.replace('Ġ', '').replace('▁', '');
        const score = attr.score;
        const className = score > 0 ? 'positive' : 'negative';
        const opacity = Math.min(Math.abs(score) * 2, 1);
        
        // Find and highlight token in text
        const regex = new RegExp(`\\b${escapeRegex(token)}\\b`, 'gi');
        highlightedHTML = highlightedHTML.replace(regex, (match) => {
            return `<span class="token-highlight ${className}" 
                          style="opacity: ${opacity}" 
                          data-score="${score.toFixed(3)}"
                          title="Attribution: ${score.toFixed(3)}">${match}</span>`;
        });
    });
    
    textWithHighlights.innerHTML = highlightedHTML;
    
    // Add tooltip functionality
    textWithHighlights.querySelectorAll('.token-highlight').forEach(span => {
        span.addEventListener('mouseenter', showTooltip);
        span.addEventListener('mouseleave', hideTooltip);
    });
    
    // Update depth slider
    updateHighlightDepth();
}

function updateHighlightDepth() {
    const depthSlider = document.getElementById('depthSlider');
    const depthValue = document.getElementById('depthValue');
    const threshold = depthSlider.value / 100;
    
    depthValue.textContent = `${depthSlider.value}%`;
    
    // Filter highlights based on threshold
    const highlights = document.querySelectorAll('.token-highlight');
    highlights.forEach(span => {
        const score = Math.abs(parseFloat(span.dataset.score));
        if (score < threshold) {
            span.style.opacity = '0.2';
        } else {
            const opacity = Math.min(score * 2, 1);
            span.style.opacity = opacity;
        }
    });
}

function updateCounterfactuals(data) {
    const counterfactualsCard = document.getElementById('counterfactualsCard');
    const counterfactualsList = document.getElementById('counterfactualsList');
    
    if (!data.counterfactuals || data.counterfactuals.length === 0) {
        counterfactualsCard.style.display = 'none';
        return;
    }
    
    counterfactualsCard.style.display = 'block';
    counterfactualsList.innerHTML = '';
    
    data.counterfactuals.forEach((cf, index) => {
        const item = document.createElement('div');
        item.className = 'counterfactual-item';
        
        const deltaSign = cf.delta_confidence > 0 ? '+' : '';
        item.innerHTML = `
            <strong>Edit ${index + 1}:</strong> "${cf.edit}"<br>
            <strong>New Label:</strong> ${cf.new_label}<br>
            <strong>Confidence Change:</strong> ${deltaSign}${(cf.delta_confidence * 100).toFixed(1)}%
        `;
        
        counterfactualsList.appendChild(item);
    });
}

function showTooltip(e) {
    const score = e.target.dataset.score;
    const tooltip = document.createElement('div');
    tooltip.className = 'token-tooltip';
    tooltip.textContent = `Score: ${score}`;
    e.target.style.position = 'relative';
    e.target.appendChild(tooltip);
}

function hideTooltip(e) {
    const tooltip = e.target.querySelector('.token-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

async function submitFeedback() {
    const activeButton = document.querySelector('.btn-feedback.active');
    const comments = document.getElementById('feedbackComments').value;
    
    if (!activeButton && !comments) {
        alert('Please provide feedback or comments.');
        return;
    }
    
    const feedbackType = activeButton ? activeButton.dataset.type : 'general';
    
    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: currentText,
                prediction: currentResults,
                feedback_type: feedbackType,
                comments: comments
            })
        });
        
        if (response.ok) {
            alert('Thank you for your feedback!');
            document.getElementById('feedbackComments').value = '';
            document.querySelectorAll('.btn-feedback').forEach(b => b.classList.remove('active'));
        } else {
            throw new Error('Failed to submit feedback');
        }
    } catch (error) {
        alert('Error submitting feedback: ' + error.message);
        console.error('Error:', error);
    }
}

function exportResults() {
    if (!currentResults) {
        alert('No results to export.');
        return;
    }
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `sentiment_analysis_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

