// JavaScript for PDF Search Interface
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const modelSelect = document.getElementById('model-select');
    const queryInput = document.getElementById('query-input');
    const searchBtn = document.getElementById('search-btn');
    const loadingDiv = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const clearResultsBtn = document.getElementById('clear-results-btn');
    const errorMessage = document.getElementById('error-message');
    const modelStatus = document.getElementById('model-status');
    const expansionCheckbox = document.getElementById('expansion-checkbox');
    const expansionInfoDiv = document.getElementById('expansion-info');
    const originalQueryText = document.getElementById('original-query-text');
    const expandedQueryText = document.getElementById('expanded-query-text');

    // Counter for search results
    let searchCounter = 0;

    // Event listeners
    searchBtn.addEventListener('click', performSearch);
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            performSearch();
        }
    });
    
    modelSelect.addEventListener('change', updateModelStatus);
    clearResultsBtn.addEventListener('click', clearAllResults);

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

        // Clear previous errors and progress messages (but keep results)
        hideError();
        hideProgress();
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
                    model: selectedModel,
                    use_expansion: expansionCheckbox.checked
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
        const { query, model, results, expansion_info, expanded_query, synthesis } = data;
        
        // Show expansion info if expansion was used
        if (expansion_info && expansion_info.expansion_used) {
            showExpansionInfo(query, expanded_query, expansion_info);
        } else {
            hideExpansionInfo();
        }
        
        // Increment search counter
        searchCounter++;

        // Create a new search result set container
        const searchResultSet = document.createElement('div');
        searchResultSet.className = 'search-result-set';
        searchResultSet.id = `search-${searchCounter}`;

        // Create search info header
        const searchInfo = document.createElement('div');
        searchInfo.className = 'search-info';
        const timestamp = new Date().toLocaleTimeString();
        
        let queryDisplayText = `"${escapeHtml(query)}"`;
        if (expansion_info && expansion_info.expansion_used) {
            queryDisplayText += ` (expanded with AI)`;
        }
        
        searchInfo.innerHTML = `
            <div class="search-info-content">
                <strong>Query:</strong> ${queryDisplayText} | 
                <strong>Model:</strong> ${escapeHtml(model)} | 
                <strong>Results:</strong> ${results.length} | 
                <strong>Time:</strong> ${timestamp}
            </div>
            <button class="btn btn-small btn-remove" onclick="removeSearchResult('${searchResultSet.id}')">
                Remove
            </button>
        `;

        // Create synthesis section if synthesis is available
        let synthesisElement = null;
        if (synthesis) {
            synthesisElement = document.createElement('div');
            synthesisElement.className = 'synthesis-section';
            synthesisElement.innerHTML = `
                <div class="synthesis-header">
                    <h3>🤖 AI Analysis</h3>
                    <span class="synthesis-subtitle">How political parties relate to your query</span>
                </div>
                <div class="synthesis-content">
                    ${escapeHtml(synthesis).replace(/\n/g, '<br>')}
                </div>
            `;
        }

        // Create results container for this search
        const searchResults = document.createElement('div');
        searchResults.className = 'search-results';

        if (results.length === 0) {
            searchResults.innerHTML = `
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
                searchResults.appendChild(resultElement);
            });
        }

        // Assemble the search result set
        searchResultSet.appendChild(searchInfo);
        
        // Add synthesis section if available
        if (synthesisElement) {
            searchResultSet.appendChild(synthesisElement);
        }
        
        searchResultSet.appendChild(searchResults);

        // Insert at the top of results container (newest first)
        resultsContainer.insertBefore(searchResultSet, resultsContainer.firstChild);

        showResults();
    }

    function createResultElement(result) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-item';
        
        // Build metadata items dynamically
        let metaItems = `
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
            </div>`;
        
        // Add new metadata fields if they exist
        if (result.type) {
            metaItems += `
            <div class="meta-item">
                <span class="meta-label">Type:</span>
                <span class="meta-badge meta-type">${escapeHtml(result.type)}</span>
            </div>`;
        }
        
        if (result.page_number) {
            metaItems += `
            <div class="meta-item">
                <span class="meta-label">Page:</span>
                <span class="meta-badge meta-page">${escapeHtml(result.page_number)}</span>
            </div>`;
        }
        
        if (result.header) {
            metaItems += `
            <div class="meta-item">
                <span class="meta-label">Section:</span>
                <span class="meta-header">${escapeHtml(result.header)}</span>
            </div>`;
        }
        
        resultDiv.innerHTML = `
            <div class="result-header">
                <div class="result-title">${result.rank}. ${escapeHtml(result.document)}</div>
                <div class="result-score">Score: ${result.similarity_score}</div>
            </div>
            
            <div class="result-meta">
                ${metaItems}
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
            modelStatus.textContent = 'Database is loaded!';
            modelStatus.className = 'model-status loaded';
        } else if (status === 'available') {
            modelStatus.textContent = 'Database exists but is not running. It will start on first search.';
            modelStatus.className = 'model-status available';
        } else if (status === 'creating') {
            modelStatus.textContent = 'Database is being created. Please wait...';
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

    function clearAllResults() {
        resultsContainer.innerHTML = '';
        searchCounter = 0;
        hideResults();
    }

    // Global function to remove individual search results
    window.removeSearchResult = function(searchId) {
        const searchElement = document.getElementById(searchId);
        if (searchElement) {
            searchElement.remove();
            
            // Hide results section if no more results
            if (resultsContainer.children.length === 0) {
                hideResults();
            }
        }
    };

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

    function showExpansionInfo(originalQuery, expandedQuery, expansionData) {
        originalQueryText.textContent = originalQuery;
        expandedQueryText.textContent = expandedQuery;
        expansionInfoDiv.classList.remove('hidden');
    }

    function hideExpansionInfo() {
        expansionInfoDiv.classList.add('hidden');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
