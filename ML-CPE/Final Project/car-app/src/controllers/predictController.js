const { spawn } = require('child_process');
const path = require('path');

const PYTHON_TIMEOUT_MS = 30_000;
const MIN_CAR_YEAR = 1886;
const MAX_CAR_YEAR = 2026;

exports.estimateCarPrice = (req, res) => {
    const carData = req.body;
    if (!carData || typeof carData !== 'object' || Array.isArray(carData)) {
        return res.status(400).json({ success: false, message: 'Car data must be a JSON object.' });
    }

    const name = typeof carData.name === 'string' ? carData.name.trim() : '';
    const year = Number(carData.year);
    const kmDriven = Number(carData.km_driven);

    if (!name) return res.status(400).json({ success: false, message: 'Car name is required.' });
    if (!Number.isInteger(year) || year < MIN_CAR_YEAR || year > MAX_CAR_YEAR) {
        return res.status(400).json({ success: false, message: `Car year must be an integer from ${MIN_CAR_YEAR} to ${MAX_CAR_YEAR}.` });
    }
    if (carData.km_driven === '' || !Number.isFinite(kmDriven) || kmDriven < 0) {
        return res.status(400).json({ success: false, message: 'Mileage must be a number greater than or equal to 0.' });
    }

    const scriptPath = path.join(__dirname, '../../classification/predict.py');
    const pythonProcess = spawn(process.env.PYTHON_PATH || 'python', [scriptPath], { windowsHide: true });
    let resultData = '';
    let errorData = '';
    let settled = false;

    const finish = (status, body) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        if (!res.headersSent) res.status(status).json(body);
    };
    const timeout = setTimeout(() => {
        pythonProcess.kill();
        finish(504, { success: false, message: 'Price estimation timed out. Please try again.' });
    }, PYTHON_TIMEOUT_MS);

    pythonProcess.stdout.on('data', (chunk) => { resultData += chunk.toString(); });
    pythonProcess.stderr.on('data', (chunk) => { errorData += chunk.toString(); });
    pythonProcess.on('error', (err) => {
        console.error('Unable to start Python prediction process:', err.message);
        finish(500, { success: false, message: 'Could not start the price estimation service.' });
    });
    pythonProcess.on('close', (code) => {
        if (settled) return;
        if (code !== 0) {
            console.error(`Python prediction failed (exit ${code}):`, errorData);
            return finish(500, { success: false, message: 'An error occurred while estimating the car price.' });
        }
        try {
            const firstBrace = resultData.indexOf('{');
            const lastBrace = resultData.lastIndexOf('}');
            if (firstBrace < 0 || lastBrace < firstBrace) throw new Error('Python returned no JSON');
            const response = JSON.parse(resultData.slice(firstBrace, lastBrace + 1));
            if (!response || response.success !== true || !Number.isFinite(Number(response.predicted_price))) {
                throw new Error('Python returned an invalid prediction response');
            }
            finish(200, response);
        } catch (err) {
            console.error('Could not parse Python prediction output:', err.message);
            finish(500, { success: false, message: 'The price estimation service returned an invalid response.' });
        }
    });
    pythonProcess.stdin.on('error', (err) => {
        console.error('Could not send data to Python prediction process:', err.message);
    });
    pythonProcess.stdin.end(JSON.stringify({ ...carData, name, year, km_driven: kmDriven }));
};
