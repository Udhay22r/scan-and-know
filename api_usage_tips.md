# API Usage Tips for Scan & Know

## Current Rate Limit Issue

You're hitting the **free tier rate limits** for the Gemini API. Here's what's happening and how to fix it:

### Free Tier Limits (Gemini API)
- **Requests per minute**: ~15 requests
- **Requests per day**: ~1500 requests  
- **Input tokens per minute**: ~15000 tokens
- **Input tokens per day**: ~1500000 tokens

### What I've Implemented to Help

1. **Rate Limiting**: Added client-side rate limiting (10 requests per minute)
2. **Retry Logic**: Automatic retry with exponential backoff
3. **Fallback Responses**: Helpful responses when API is unavailable
4. **Model Prioritization**: Uses `gemini-1.5-flash` first (lower rate limits)

### Immediate Solutions

#### Option 1: Wait and Retry
- Wait 1-2 minutes before trying again
- The rate limits reset automatically

#### Option 2: Upgrade Your API Plan
1. Go to https://makersuite.google.com/app/apikey
2. Click on your API key
3. Upgrade to a paid plan for higher limits

#### Option 3: Use the App Without AI Chat
- The barcode scanning and product analysis still work perfectly
- Only the AI chatbot is affected by rate limits

### Best Practices

1. **Don't spam the chatbot** - Wait between questions
2. **Use specific questions** - Shorter prompts use fewer tokens
3. **Check product analysis first** - Most info is already displayed
4. **Use during off-peak hours** - Less competition for API resources

### Monitoring Your Usage

You can check your API usage at:
https://makersuite.google.com/app/apikey

### Alternative Solutions

If you continue having issues:

1. **Use a different API key** - Create a new one
2. **Implement caching** - Store common responses
3. **Use a different AI service** - Consider OpenAI or other alternatives
4. **Go offline-only** - Remove AI features temporarily

### Current Status

✅ **Fixed**: Rate limiting and retry logic implemented  
✅ **Fixed**: Fallback responses when API is unavailable  
✅ **Fixed**: Better error messages for users  
⚠️ **Note**: You may still hit limits during heavy usage 