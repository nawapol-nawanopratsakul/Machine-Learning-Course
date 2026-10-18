const express = require('express');
const cors = require('cors');
const path = require('path');
const predictController = require('./controllers/predictController');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public'))); // ให้บริการไฟล์ Frontend Static

// API Route สำหรับประเมินราคา
app.post('/api/predict', predictController.estimateCarPrice);

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});