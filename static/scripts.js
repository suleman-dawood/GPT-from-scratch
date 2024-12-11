// Function to call the backend API
async function generateOutput() {
    const button = document.getElementById("generate-button");
    const outputArea = document.getElementById("output-text");
    button.disabled = true;
    button.textContent = "Generating...";

    try {
        // Call the backend endpoint
        const response = await fetch('/generate', { method: 'POST' });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Get the plain text result and log it to the console
        const result = await response.text();
        console.log(result);  // Log the result in the console

        // Also display the result in the textarea
        outputArea.value = result;  // Since textarea preserves newlines, this should work

    } catch (error) {
        console.error(`An error occurred: ${error.message}`);  // Log errors in the console
        outputArea.value = `An error occurred:\n${error.message}`; // Display error in textarea
    } finally {
        button.disabled = false;
        button.textContent = "Generate";
    }
}

// Attach event listener to the button
document.getElementById("generate-button").addEventListener("click", generateOutput);
