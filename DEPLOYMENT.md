# Deployment Guide for StockForecastX Pro

This application requires **two separate deployments**:
1. **Frontend** (HTML/CSS/JS) → Netlify
2. **Backend** (Flask API) → Railway, Render, or Heroku

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Netlify       │  HTTP   │  Flask Backend    │
│  (Frontend)     │────────▶│  (Railway/Render) │
│  index.html     │         │  app.py           │
│  styles.css     │         │  API Endpoints    │
│  script.js      │         │  ML Models        │
└─────────────────┘         └──────────────────┘
```

## Step 1: Deploy Flask Backend

### Option A: Railway (Recommended - Free tier available)

1. **Sign up** at [railway.app](https://railway.app)
2. **Create a new project**
3. **Connect your GitHub repository**
4. **Add a new service** → Select your `Stock_DEV` directory
5. **Configure the service:**
   - **Start Command**: `python3 app.py`
   - **Environment Variables** (add these in Railway dashboard):
     ```
     ALPHA_VANTAGE_API_KEY=your_key_here
     OPENAI_API_KEY=your_key_here
     OPENWEATHER_API_KEY=your_key_here
     PORT=5000
     ```
6. **Deploy** - Railway will automatically deploy
7. **Copy your Railway URL** (e.g., `https://your-app.railway.app`)

### Option B: Render (Free tier available)

1. **Sign up** at [render.com](https://render.com)
2. **Create a new Web Service**
3. **Connect your GitHub repository**
4. **Configure:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: Add your API keys
5. **Deploy** and copy your Render URL

### Option C: Heroku

1. **Install Heroku CLI**
2. **Login**: `heroku login`
3. **Create app**: `heroku create your-app-name`
4. **Set environment variables**:
   ```bash
   heroku config:set ALPHA_VANTAGE_API_KEY=your_key
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set OPENWEATHER_API_KEY=your_key
   ```
5. **Deploy**: `git push heroku main`

## Step 2: Update Frontend API URL

After deploying your backend, update `script.js`:

```javascript
// Replace this line in script.js:
const prodUrl = window.API_BASE_URL || 'https://your-flask-backend.railway.app/api';
// With your actual backend URL:
const prodUrl = window.API_BASE_URL || 'https://your-actual-backend-url.com/api';
```

## Step 3: Deploy Frontend to Netlify

### Method 1: Via Netlify Dashboard

1. **Sign up** at [netlify.com](https://netlify.com)
2. **Add new site** → **Import from Git**
3. **Connect your GitHub repository**
4. **Configure build settings:**
   - **Build command**: (leave empty - no build needed)
   - **Publish directory**: `.` (root)
5. **Deploy site**

### Method 2: Via Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod
```

### Configure Environment Variable (Optional)

In Netlify dashboard:
1. Go to **Site settings** → **Environment variables**
2. Add: `API_BASE_URL` = `https://your-backend-url.com/api`

This will be available as `window.API_BASE_URL` in your JavaScript.

## Step 4: Update CORS Settings

Make sure your Flask backend allows requests from your Netlify domain:

In `app.py`, update the CORS configuration:

```python
from flask_cors import CORS

# Allow requests from Netlify domain
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://your-netlify-site.netlify.app",
            "http://localhost:5004"  # Keep for local dev
        ]
    }
})
```

Or allow all origins (less secure, but easier for testing):

```python
CORS(app)  # Allows all origins
```

## Testing

1. **Local Development**: 
   - Frontend: Open `index.html` in browser or use `python3 -m http.server`
   - Backend: Run `python3 app.py` on port 5004

2. **Production**:
   - Visit your Netlify URL
   - The frontend will automatically connect to your deployed backend

## Troubleshooting

### CORS Errors
- Make sure CORS is configured in `app.py`
- Check that your Netlify domain is allowed

### API Connection Failed
- Verify your backend URL is correct in `script.js`
- Check that your backend is running and accessible
- Test backend directly: `curl https://your-backend-url.com/api/test`

### Environment Variables Not Working
- Double-check variable names match exactly
- Restart your backend service after adding variables
- Check backend logs for errors

## Quick Reference

- **Frontend URL**: `https://your-site.netlify.app`
- **Backend URL**: `https://your-backend.railway.app` (or Render/Heroku)
- **API Endpoint**: `https://your-backend.railway.app/api/analyze`

