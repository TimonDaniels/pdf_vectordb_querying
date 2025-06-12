// JavaScript for PDF Search Interface
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const modelSelect = document.getElementById('model-select');
    const queryInput = document.getElementById('query-input');
    const searchBtn = document.getElementById('search-btn');
    const loadingDiv = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const searchInfo = document.getElementById('search-info');
    const errorMessage = document.getElementById('error-message');
    const modelStatus = document.getElementById('model-status');

    // Event listeners
    searchBtn.addEventListener('click', performSearch);
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            performSearch();
        }
    });
    
    modelSelect.addEventListener('change', updateModelStatus);

    // Auto-focus on query input
    queryInput.focus();
    
    // Update initial model status
    updateModelStatus();
    
    // Poll for model status updates every 5 seconds
    setInterval(refreshModelStatus, 5000);

    async function performSearch() {
        const query = queryInput.value.trim();
        const selectedModel = modelSelect.value;

        // Validation
        if (!query) {
            showError('Please enter a search query.');
            return;
        }

        if (!selectedModel) {
            showError('Please select a database model.');
            return;
        }

        // Clear previous results, errors, and progress messages
        hideError();
        hideProgress();
        hideResults();
        showLoading();
        
        // Disable search button
        searchBtn.disabled = true;
        searchBtn.textContent = 'Searching...';

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    model: selectedModel
                })
            });

            const data = await response.json();

            if (response.status === 202) {
                // Database is being created or needs to be created
                showProgress(data.message || data.error); // Support both new and old message format
                // Trigger model status refresh
                setTimeout(refreshModelStatus, 1000);
                return;
            }

            if (!response.ok) {
                throw new Error(data.error || 'Search failed');
            }

            displayResults(data);

        } catch (error) {
            console.error('Search error:', error);
            showError(`Search failed: ${error.message}`);
        } finally {
            hideLoading();
            // Re-enable search button
            searchBtn.disabled = false;
            searchBtn.textContent = 'Search';
        }
    }

    function displayResults(data) {
        const { query, model, results } = data;

        // Update search info
        searchInfo.innerHTML = `
            <strong>Query:</strong> "${query}" | 
            <strong>Model:</strong> ${model} | 
            <strong>Results:</strong> ${results.length}
        `;

        // Clear previous results
        resultsContainer.innerHTML = '';

        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="result-item">
                    <p style="text-align: center; color: #7f8c8d; font-style: italic;">
                        No relevant content found for your query.
                    </p>
                </div>
            `;
        } else {
            // Display each result
            results.forEach(result => {
                const resultElement = createResultElement(result);
                resultsContainer.appendChild(resultElement);
            });
        }

        showResults();
    }

    function createResultElement(result) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-item';
        
        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="result-title">${result.rank}. ${escapeHtml(result.document)}</div>
                <div class="result-score">Score: ${result.similarity_score}</div>
            </div>
            
            <div class="result-meta">
                <div class="meta-item">
                    <span class="meta-label">File:</span>
                    <span>${escapeHtml(result.filename)}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Chunk ID:</span>
                    <span>${escapeHtml(result.chunk_id)}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Model:</span>
                    <span>${escapeHtml(result.embedding_model)}</span>
                </div>
            </div>
            
            <div class="result-content">
                ${escapeHtml(result.content)}
            </div>
        `;

        return resultDiv;
    }

    async function updateModelStatus() {
        const selectedModel = modelSelect.value;
        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        
        if (!selectedModel || !selectedOption) {
            modelStatus.style.display = 'none';
            return;
        }

        const status = selectedOption.dataset.status;
        
        if (status === 'loaded') {
            modelStatus.textContent = 'Database is loaded in memory - ready for fast searching!';
            modelStatus.className = 'model-status loaded';
        } else if (status === 'available') {
            modelStatus.textContent = 'Database is ready. Will be loaded into memory on first search.';
            modelStatus.className = 'model-status available';
        } else if (status === 'creating') {
            modelStatus.textContent = 'Database is being created in the background. Please wait...';
            modelStatus.className = 'model-status creating';
        } else if (status === 'failed') {
            modelStatus.textContent = 'Database creation failed. Please try refreshing the page.';
            modelStatus.className = 'model-status failed';
        } else if (status === 'not_created') {
            modelStatus.textContent = 'Database needs to be created. It will be created automatically when you search.';
            modelStatus.className = 'model-status not-created';
        } else {
            modelStatus.style.display = 'none';
        }
    }

    async function refreshModelStatus() {
        try {
            const response = await fetch('/models');
            const data = await response.json();
            
            if (data.models && data.models.length > 0) {
                const currentValue = modelSelect.value;
                
                // Update the select options with new status
                data.models.forEach(model => {
                    const option = modelSelect.querySelector(`option[value="${model.name}"]`);
                    if (option) {
                        option.dataset.status = model.status;
                        
                        // Update option text
                        let statusText = '';
                        if (model.status === 'loaded') {
                            statusText = ' 🚀 Loaded';
                        } else if (model.status === 'available') {
                            statusText = ' ✓ Available';
                        } else if (model.status === 'creating') {
                            statusText = ' ⏳ Creating...';
                        } else if (model.status === 'failed') {
                            statusText = ' ✗ Failed';
                        } else {
                            statusText = ' (Not created)';
                        }
                        
                        option.textContent = model.name + statusText;
                    }
                });
                
                // Update status for currently selected model
                updateModelStatus();
            }
        } catch (error) {
            console.error('Failed to refresh model status:', error);
        }
    }

    function showLoading() {
        loadingDiv.classList.remove('hidden');
    }

    function hideLoading() {
        loadingDiv.classList.add('hidden');
    }

    function showResults() {
        resultsSection.classList.remove('hidden');
        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function hideResults() {
        resultsSection.classList.add('hidden');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
        // Hide progress message if it's showing
        hideProgress();
        // Scroll to error message
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }

    function showProgress(message) {
        const progressMessage = document.getElementById('progress-message');
        progressMessage.textContent = message;
        progressMessage.classList.remove('hidden');
        // Hide error message if it's showing
        hideError();
        // Scroll to progress message
        progressMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideProgress() {
        const progressMessage = document.getElementById('progress-message');
        progressMessage.classList.add('hidden');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
