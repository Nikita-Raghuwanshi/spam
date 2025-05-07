const express = require('express');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');
const cors = require('cors'); // Import CORS

const app = express();
const port = 3001;

// Enable CORS for all routes
app.use(cors());

// Middleware to parse JSON request bodies
app.use(bodyParser.json());

// Endpoint to predict spam or not
app.post('/predict', (req, res) => {
    const emailText = req.body.email_text;

    let options = {
        mode: 'text',
        pythonPath: 'python',  // Ensure Python is installed and in your system's PATH. If using python3, update it to 'python3'
        pythonOptions: ['-u'], // Unbuffered output
        scriptPath: './',  // Path to your Python script (adjust the folder if necessary)
        args: [emailText]  // Pass the email text as an argument to the Python script
    };

    // Run the Python script (predict.py instead of main1.py)
    PythonShell.run('predict.py', options, (err, result) => {
        if (err) {
            console.error("Error running Python script", err);
            res.status(500).json({ error: 'Error in prediction' });
        } else {
            res.json({ prediction: result[0] });  // Return the result from Python script
        }
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
