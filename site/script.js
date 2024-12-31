document.addEventListener('DOMContentLoaded', function () {
    const featuresInput = document.getElementById('features');
    const predictButton = document.getElementById('predict-button');
    const errorMessage = document.getElementById('error-message');
    const predictionResults = document.getElementById('prediction-results');
    const predictionOutput = document.getElementById('prediction-output');

    let loading = false;
    function jsonifyWithoutQuotes(obj) {
        let jsonString = JSON.stringify(obj, null, 2);
        return jsonString
            .replace(/"([^"]+)":/g, '$1:')  // Removes quotes around keys
            .replace(/"([^"]+)"/g, '$1');   // Removes quotes around string values
    }


    async function handleSubmit() {
        if (featuresInput.value.length < 2000) {
            alert("Text is too short for detector! Please enter at least 2000 characters.");
            return;
        }

        if (loading) return;
        loading = true;

        try {
            errorMessage.textContent = '';
            predictionResults.style.display = 'none';
            predictButton.textContent = 'Processing...';

            const response = await fetch('https://api-239872853943.europe-central2.run.app/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: featuresInput.value.trim()
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            predictionOutput.textContent = jsonifyWithoutQuotes(data["text_result"])
            predictionResults.style.display = 'block';
            if (data["result"]) {
                predictionOutput.style.color = "red";
            } else {
                predictionOutput.style.color = "green"
            }
        } catch (err) {
            errorMessage.textContent = 'Error connecting to API';
        } finally {
            loading = false;
            predictButton.textContent = 'Predict';
        }
    }

    predictButton.addEventListener('click', handleSubmit);
});
