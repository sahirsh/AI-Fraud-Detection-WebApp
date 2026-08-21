import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {

  const [jobText, setJobText] = useState(''); // Stores user input
  const [result, setResult] = useState(null); //stores API response
  const [loading, setLoading] = useState(false); // Loading State
  const [error, setError] = useState(null); // errors

  const checkFraud = async () => {
    setLoading(true);
    setError(null);
    
    try {
      if (!jobText.trim()) {
        setError("Please paste a job posting first.");
        return;
      }

      const response = await axios.post('https://ai-fraud-detection-webapp-mdcc.onrender.com/predict', {
        job_posting: jobText
      });

      setResult(response.data);
    }catch (err) {
      console.error(err);
      setError('Error calling the API');

    }finally{
      setLoading(false);
    }

  };

  return (
  <div className="App" style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
    <h1>Job Posting Fraud Detector</h1>

    <textarea
      rows={8}
      cols={50}
      placeholder="Paste job posting text here..."
      value={jobText}
      onChange={(e) => setJobText(e.target.value)}
    />

    <br />
    <button onClick={checkFraud} disabled={loading} style={{ marginTop: '1rem' }}>
      {loading ? 'Checking...' : 'Check Fraud'}
    </button>

    {error && <p style={{ color: 'red' }}>{error}</p>}

    {result && (
      <div style={{ marginTop: '1rem', border: '1px solid gray', padding: '1rem' }}>
        <p><strong>Fraud Probability:</strong> {result.fraud_probability}</p>
        <p><strong>Label:</strong> {result.label}</p>
        <p><strong>Risk Level:</strong> {result.risk_level}</p>
      </div>
    )}
  </div>
);

}

export default App
