# Quick Guide: Deploy Backend to Railway

Your frontend is live on Netlify, but you need to deploy the Flask backend separately. Here's how:

## Step 1: Deploy to Railway

1. **Sign up** at [railway.app](https://railway.app) (free tier available)

2. **Create New Project** → **Deploy from GitHub repo**
   - Select your repository: `Stock-Pred-with-ml`
   - Railway will detect it's a Python project

3. **Add Environment Variables** (in Railway dashboard → Variables tab):
   ```
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
   OPENAI_API_KEY=your_openai_key_here
   OPENWEATHER_API_KEY=your_openweather_key_here
   PORT=5000
   ```
   
   **Get your API keys from:**
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key
   - OpenAI: https://platform.openai.com/api-keys
   - OpenWeather: https://openweathermap.org/api

4. **Configure Service**:
   - **Start Command**: `python3 app.py` (or Railway will auto-detect from Procfile)
   - Railway will automatically detect Python version from your `requirements.txt` and `Procfile`
   - Railway defaults to Python 3.11, which works perfectly for this app

5. **Deploy** - Railway will build and deploy automatically

6. **Get Your Backend URL**:
   - After deployment, Railway will give you a URL like: `https://your-app-name.railway.app`
   - Copy this URL!

## Step 2: Update Frontend with Backend URL

1. **Update `script.js`** line 9:
   ```javascript
   const prodUrl = window.API_BASE_URL || 'https://YOUR-ACTUAL-RAILWAY-URL.railway.app/api';
   ```
   Replace `YOUR-ACTUAL-RAILWAY-URL` with your Railway URL (without the `https://` and `.railway.app` parts)

2. **Commit and push**:
   ```bash
   git add script.js
   git commit -m "Update backend URL for production"
   git push origin main
   ```

3. **Netlify will automatically redeploy** with the new backend URL

## Alternative: Use Netlify Environment Variable

Instead of hardcoding the URL, you can set it in Netlify:

1. Go to Netlify Dashboard → Site Settings → Environment Variables
2. Add: `API_BASE_URL` = `https://your-railway-url.railway.app/api`
3. Update `script.js` to read from `window.API_BASE_URL` (already configured!)

## Troubleshooting

- **Backend not starting?** Check Railway logs for errors
- **CORS errors?** Make sure CORS in `app.py` allows your Netlify domain
- **API keys not working?** Double-check environment variables in Railway

