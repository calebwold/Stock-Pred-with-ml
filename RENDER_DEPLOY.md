# Deploy Backend to Render

Your frontend is live on Netlify, now let's deploy the Flask backend to Render.

## Step 1: Sign Up for Render

1. Go to [render.com](https://render.com) and sign up (free tier available)
2. Connect your GitHub account

## Step 2: Create a New Web Service

1. In Render dashboard, click **"New +"** → **"Web Service"**
2. Connect your GitHub repository: `Stock-Pred-with-ml`
3. Render will auto-detect it's a Python project

## Step 3: Configure the Service

### Basic Settings:
- **Name**: `stockforecastx-backend` (or any name you prefer)
- **Region**: Choose closest to you (e.g., `Oregon (US West)`)
- **Branch**: `main`
- **Root Directory**: Leave empty (or `.` if needed)

### Build & Start Commands:
- **Build Command**: 
  ```
  pip install -r requirements.txt
  ```
- **Start Command**: 
  ```
  python3 app.py
  ```
  (Render will automatically set the `PORT` environment variable)

### Environment Variables:
Click **"Add Environment Variable"** and add:

```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
OPENAI_API_KEY=your_openai_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
```

**Get your API keys from:**
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- OpenAI: https://platform.openai.com/api-keys
- OpenWeather: https://openweathermap.org/api

### Advanced Settings (Optional):
- **Auto-Deploy**: `Yes` (deploys automatically on git push)
- **Health Check Path**: `/api/test` (if you have a test endpoint)

## Step 4: Deploy

1. Click **"Create Web Service"**
2. Render will:
   - Install Python and dependencies
   - Build your application
   - Start the Flask server
3. Wait for deployment to complete (5-10 minutes first time)

## Step 5: Get Your Backend URL

After deployment succeeds, Render will give you a URL like:
```
https://stockforecastx-backend.onrender.com
```

**Important**: Render free tier services spin down after 15 minutes of inactivity. The first request after spin-down may take 30-60 seconds to wake up.

## Step 6: Update Frontend with Backend URL

1. **Update `script.js`** line 9:
   ```javascript
   const prodUrl = window.API_BASE_URL || 'https://YOUR-RENDER-APP-NAME.onrender.com/api';
   ```
   Replace `YOUR-RENDER-APP-NAME` with your actual Render service name

2. **Commit and push**:
   ```bash
   git add script.js
   git commit -m "Update backend URL for Render deployment"
   git push origin main
   ```

3. **Netlify will automatically redeploy** with the new backend URL

## Alternative: Use Netlify Environment Variable

Instead of hardcoding the URL, you can set it in Netlify:

1. Go to Netlify Dashboard → Site Settings → Environment Variables
2. Add: `API_BASE_URL` = `https://your-render-app.onrender.com/api`
3. The `script.js` already reads from `window.API_BASE_URL`!

## Troubleshooting

### Backend not starting?
- Check Render logs for errors
- Verify all environment variables are set correctly
- Make sure `requirements.txt` has all dependencies

### CORS errors?
- Your `app.py` already has CORS configured to allow all origins
- If you want to restrict it, update line 28 in `app.py` to your Netlify domain

### Slow first request?
- This is normal on Render free tier (services spin down after inactivity)
- Consider upgrading to paid tier for always-on service

### Port errors?
- Render automatically sets the `PORT` environment variable
- Your `app.py` already reads it: `port = int(os.getenv('PORT', 5004))`

## Render Free Tier Limits

- Services spin down after 15 minutes of inactivity
- 750 hours/month free (enough for always-on if you use it regularly)
- First request after spin-down takes 30-60 seconds

## Next Steps

Once deployed:
1. Test your backend: Visit `https://your-app.onrender.com/api/test` (if you have a test endpoint)
2. Update frontend URL in `script.js`
3. Test the full application on Netlify!

