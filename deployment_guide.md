# Deployment Guide for Scan & Know

## 🚨 Security Considerations Before Deployment

### 1. API Key Security
- **NEVER commit your `.env` file to version control**
- Use environment variables on your hosting platform
- Consider using a separate API key for production

### 2. Rate Limiting
- Current limit: 10 requests/minute per user
- Consider upgrading to a paid Gemini API plan
- Implement user authentication to prevent abuse

### 3. Resource Management
- OCR processing is CPU-intensive
- Consider using a VPS or dedicated server
- Monitor memory and CPU usage

## 🌐 Hosting Options

### Option 1: Heroku (Recommended for Beginners)
**Pros:**
- Easy deployment
- Free tier available
- Good documentation

**Cons:**
- Requires buildpacks for Tesseract
- Limited resources on free tier

**Setup:**
1. Create `Procfile`:
```
web: python app.py
```

2. Create `runtime.txt`:
```
python-3.11.0
```

3. Add buildpack for Tesseract:
```bash
heroku buildpacks:add https://github.com/heroku/heroku-buildpack-apt
```

4. Create `Aptfile`:
```
tesseract-ocr
```

### Option 2: Railway
**Pros:**
- Easy deployment
- Good for Python apps
- Automatic HTTPS

**Setup:**
1. Connect your GitHub repository
2. Set environment variables in Railway dashboard
3. Deploy automatically

### Option 3: DigitalOcean App Platform
**Pros:**
- More control
- Better performance
- Scalable

**Cons:**
- Requires more setup
- Paid service

### Option 4: VPS (DigitalOcean, AWS, etc.)
**Pros:**
- Full control
- Best performance
- Can handle multiple users

**Cons:**
- Requires server management
- More complex setup

## 🔧 Required Changes for Deployment

### 1. Environment Variables
Replace hardcoded API key with environment variable:

```python
# In app.py, change:
api_key = os.getenv('GEMINI_API_KEY')

# Set in hosting platform:
GEMINI_API_KEY=your_api_key_here
```

### 2. Port Configuration
Update app.py for production:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### 3. CORS Configuration
Update CORS for your domain:

```python
CORS(app, origins=['https://yourdomain.com'])
```

### 4. Error Handling
Add better error handling for production:

```python
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500
```

## 📋 Deployment Checklist

- [ ] Remove `.env` file from repository
- [ ] Set environment variables on hosting platform
- [ ] Update CORS settings
- [ ] Test OCR functionality on server
- [ ] Monitor API usage
- [ ] Set up logging
- [ ] Configure HTTPS
- [ ] Test with multiple users

## 🔒 Security Best Practices

1. **API Key Management:**
   - Use environment variables
   - Rotate keys regularly
   - Monitor usage

2. **Rate Limiting:**
   - Implement user authentication
   - Add request throttling
   - Monitor abuse

3. **Input Validation:**
   - Validate uploaded images
   - Limit file sizes
   - Sanitize user inputs

4. **Error Handling:**
   - Don't expose sensitive information in errors
   - Log errors properly
   - Provide user-friendly error messages

## 📊 Monitoring and Maintenance

1. **API Usage Monitoring:**
   - Track Gemini API calls
   - Monitor rate limits
   - Set up alerts

2. **Performance Monitoring:**
   - Monitor response times
   - Track memory usage
   - Monitor CPU usage

3. **User Analytics:**
   - Track feature usage
   - Monitor error rates
   - Gather user feedback

## 🚀 Quick Deployment Steps

1. **Prepare Repository:**
   ```bash
   # Remove sensitive files
   echo ".env" >> .gitignore
   echo "venv/" >> .gitignore
   echo "__pycache__/" >> .gitignore
   ```

2. **Choose Hosting Platform:**
   - Heroku: Good for beginners
   - Railway: Easy deployment
   - VPS: Best performance

3. **Deploy:**
   - Connect repository
   - Set environment variables
   - Deploy and test

4. **Monitor:**
   - Check logs
   - Monitor API usage
   - Test functionality

## 💡 Recommendations

1. **Start Small:** Begin with a free tier to test
2. **Monitor Usage:** Keep track of API calls and costs
3. **Scale Gradually:** Upgrade as needed
4. **Backup Data:** Regular backups of user preferences
5. **User Feedback:** Gather feedback to improve

## 🆘 Troubleshooting

### Common Issues:
1. **Tesseract not found:** Install buildpack or use Docker
2. **API key errors:** Check environment variables
3. **Memory issues:** Optimize image processing
4. **Rate limiting:** Implement caching or upgrade plan

### Support:
- Check hosting platform documentation
- Monitor application logs
- Test locally before deploying 