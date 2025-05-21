$(document).ready(function() {
    // Single model prediction form
    $('#predictionForm').on('submit', function(e) {
        e.preventDefault();
        
        // Show results section and loading spinner
        $('#results').removeClass('d-none');
        $('#loadingSpinner').removeClass('d-none');
        $('#predictionResult').addClass('d-none');
        $('#comparisonResult').addClass('d-none');
        
        // Update result header
        $('#resultHeader h5').text('Price Prediction Results');
        
        // Scroll to results section
        $('html, body').animate({
            scrollTop: $('#results').offset().top - 100
        }, 500);
        
        // Get form data
        const formData = new FormData(this);
        const crypto = formData.get('cryptocurrency');
        const modelType = formData.get('model_type');
        
        // Make prediction request
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Hide loading spinner and show prediction result
                $('#loadingSpinner').addClass('d-none');
                $('#predictionResult').removeClass('d-none');
                
                // Update result details
                const details = `
                    <h4>${response.cryptocurrency} (${response.symbol}) Price Prediction</h4>
                    <p class="text-muted">Using ${modelType.toUpperCase()} model</p>
                    <div class="card mb-3">
                        <div class="card-body">
                            <p class="mb-2"><strong>Current Price:</strong> $${response.current_price.toFixed(2)}</p>
                            <p class="mb-2"><strong>Predicted Price:</strong> $${response.predicted_price.toFixed(2)}</p>
                            <p class="mb-2">
                                <strong>Price Change:</strong> 
                                $${response.price_change.toFixed(2)} 
                                <span class="${response.price_change >= 0 ? 'text-success' : 'text-danger'}">
                                    (${response.price_change_percent.toFixed(2)}%)
                                </span>
                            </p>
                            <p class="mb-0"><strong>Prediction Time:</strong> ${response.prediction_time}</p>
                        </div>
                    </div>
                `;
                
                $('#resultDetails').html(details);
                
                // Display prediction chart
                const chartHtml = `
                    <div class="card">
                        <div class="card-body">
                            <img src="/${response.plot_path}" class="img-fluid" alt="Prediction Chart">
                        </div>
                    </div>
                `;
                
                $('#predictionChart').html(chartHtml);
            },
            error: function(error) {
                // Hide loading spinner and show error message
                $('#loadingSpinner').addClass('d-none');
                $('#predictionResult').removeClass('d-none');
                
                const errorMsg = error.responseJSON && error.responseJSON.error 
                    ? error.responseJSON.error 
                    : 'An unexpected error occurred. Please try again.';
                
                $('#resultDetails').html(`<div class="alert alert-danger">${errorMsg}</div>`);
                $('#predictionChart').html('');
            }
        });
    });
    
    // Model comparison form
    $('#comparisonForm').on('submit', function(e) {
        e.preventDefault();
        
        // Show results section and loading spinner
        $('#results').removeClass('d-none');
        $('#loadingSpinner').removeClass('d-none');
        $('#predictionResult').addClass('d-none');
        $('#comparisonResult').addClass('d-none');
        
        // Update result header
        $('#resultHeader h5').text('Model Comparison Results');
        
        // Scroll to results section
        $('html, body').animate({
            scrollTop: $('#results').offset().top - 100
        }, 500);
        
        // Get form data
        const formData = new FormData(this);
        const crypto = formData.get('cryptocurrency');
        
        // Make comparison request
        $.ajax({
            url: '/compare',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Hide loading spinner and show comparison result
                $('#loadingSpinner').addClass('d-none');
                $('#comparisonResult').removeClass('d-none');
                
                // Update comparison details
                const details = `
                    <h4>${response.cryptocurrency} (${response.symbol}) Model Comparison</h4>
                    <div class="card mb-3">
                        <div class="card-body">
                            <p class="mb-2"><strong>Current Price:</strong> $${response.current_price.toFixed(2)}</p>
                            
                            <h5 class="mt-3">Model Predictions</h5>
                            
                            <div class="mb-2">
                                <strong>CNN Model:</strong> $${response.cnn.predicted_price.toFixed(2)}
                                <span class="${response.cnn.price_change >= 0 ? 'text-success' : 'text-danger'}">
                                    (${response.cnn.price_change_percent.toFixed(2)}%)
                                </span>
                            </div>
                            
                            <div class="mb-2">
                                <strong>LGBM Model:</strong> $${response.lgbm.predicted_price.toFixed(2)}
                                <span class="${response.lgbm.price_change >= 0 ? 'text-success' : 'text-danger'}">
                                    (${response.lgbm.price_change_percent.toFixed(2)}%)
                                </span>
                            </div>
                            
                            <div class="mb-2">
                                <strong>Hybrid Model:</strong> $${response.hybrid.predicted_price.toFixed(2)}
                                <span class="${response.hybrid.price_change >= 0 ? 'text-success' : 'text-danger'}">
                                    (${response.hybrid.price_change_percent.toFixed(2)}%)
                                </span>
                            </div>
                            
                            <p class="mb-0 mt-2"><strong>Prediction Time:</strong> ${response.prediction_time}</p>
                        </div>
                    </div>
                `;
                
                $('#comparisonDetails').html(details);
                
                // Display comparison chart
                const chartHtml = `
                    <div class="card">
                        <div class="card-body">
                            <img src="/${response.comparison_plot}" class="img-fluid" alt="Model Comparison Chart">
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header bg-primary text-white">CNN Model</div>
                                <div class="card-body p-0">
                                    <img src="/${response.cnn.plot_path}" class="img-fluid" alt="CNN Prediction">
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header bg-success text-white">LGBM Model</div>
                                <div class="card-body p-0">
                                    <img src="/${response.lgbm.plot_path}" class="img-fluid" alt="LGBM Prediction">
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header bg-info text-white">Hybrid Model</div>
                                <div class="card-body p-0">
                                    <img src="/${response.hybrid.plot_path}" class="img-fluid" alt="Hybrid Prediction">
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                $('#comparisonChart').html(chartHtml);
            },
            error: function(error) {
                // Hide loading spinner and show error message
                $('#loadingSpinner').addClass('d-none');
                $('#comparisonResult').removeClass('d-none');
                
                const errorMsg = error.responseJSON && error.responseJSON.error 
                    ? error.responseJSON.error 
                    : 'An unexpected error occurred. Please try again.';
                
                $('#comparisonDetails').html(`<div class="alert alert-danger">${errorMsg}</div>`);
                $('#comparisonChart').html('');
            }
        });
    });
});
