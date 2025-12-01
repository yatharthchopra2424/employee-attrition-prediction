# 🤖 NVIDIA AI Chatbot for Employee Retention

## Quick Start Guide

### 1. Get Your Free NVIDIA API Key

1. Visit [NVIDIA AI Platform](https://build.nvidia.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (keep it secure!)

### 2. Setup Configuration

```powershell
# Copy the environment template
cp .env.chatbot .env

# Edit .env and add your API key
# Replace 'your_nvidia_api_key_here' with your actual key
```

Or set it directly in PowerShell:
```powershell
$env:NVIDIA_API_KEY="your_actual_nvidia_api_key_here"
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

Key packages installed:
- `openai==1.6.1` - NVIDIA API client (compatible with OpenAI SDK)
- `streamlit==1.29.0` - Web interface
- All ML dependencies (scikit-learn, xgboost, etc.)

### 4. Launch the Chatbot

```powershell
streamlit run app_chatbot.py
```

The chatbot will open in your browser at `http://localhost:8501`

---

## 🎯 Features

### 1. **Automatic ML Predictions**
Simply mention any employee ID in your message:
- "What's the risk for employee 150?"
- "Analyze employee 25"
- "Tell me about employee 500"

The chatbot will automatically:
1. Extract the employee ID from your message
2. Load employee data from the dataset
3. Run ML prediction on the attrition model
4. Display risk level with color coding
5. Show top 5 risk factors
6. Provide AI-powered recommendations

### 2. **Streaming Responses**
Real-time AI responses powered by NVIDIA's Llama 3.3 Nemotron (49B parameters):
- Responses appear word-by-word as they're generated
- No waiting for complete responses
- Natural conversation flow

### 3. **Conversation History**
All conversations are stored and maintained:
- Full chat history displayed in the interface
- Context-aware responses based on previous messages
- Save/load conversations as JSON files
- Export conversations for record-keeping

### 4. **Intelligent Context**
The chatbot understands context:
- Remembers previous predictions in the conversation
- References earlier discussions
- Builds on prior questions
- Maintains conversation coherence

### 5. **Risk Analysis**
Comprehensive risk assessment:
- **High Risk** (≥70%): 🔴 Red indicator
- **Medium Risk** (40-69%): 🟡 Yellow indicator  
- **Low Risk** (<40%): 🟢 Green indicator

Automatic identification of:
- Poor work-life balance
- Excessive overtime
- Low job satisfaction
- Lack of promotions
- Weak manager relationships
- High workload
- Insufficient training

---

## 💬 Example Conversations

### Example 1: Simple Prediction Query
**You:** Analyze employee 150

**Chatbot:**
```
🎯 Attrition Risk Assessment
Employee 150
🔴
HIGH RISK
78.5%

⚠️ Key Risk Factors:
• Poor work-life balance
• Frequent overtime work
• Long time since last promotion
• High weekly hours (55)
• Low job satisfaction

[AI Response follows with detailed recommendations...]
```

### Example 2: General Question
**You:** What are the main causes of employee attrition?

**Chatbot:** Based on our ML model and historical data, the main causes are:

1. **Work-Life Balance**: Employees reporting poor work-life balance are 3x more likely...
2. **Overtime**: Frequent overtime correlates strongly with attrition...
[etc.]

### Example 3: Follow-up Questions
**You:** Analyze employee 200

**Chatbot:** [Shows prediction: Medium Risk 55%]

**You:** What specific actions should we take?

**Chatbot:** Based on Employee 200's Medium Risk profile, I recommend:
1. Schedule 1:1 meeting within next week...
2. Review current project load...
[etc.]

---

## 🎨 User Interface

### Main Components

1. **Sidebar (Left)**
   - API key configuration
   - Chatbot status indicator
   - Session statistics
   - Model information
   - Help section

2. **Main Chat Area**
   - Conversation history with styled messages
   - User messages (purple, right-aligned)
   - AI messages (gray, left-aligned)
   - Prediction cards (color-coded by risk)
   - Timestamps for all messages

3. **Quick Actions Bar**
   - Show Statistics
   - Clear History
   - Save Conversation
   - Suggested prompts (clickable)

4. **Input Area**
   - Text input for messages
   - Send button
   - Auto-clear after sending

### Visual Design
- **Gradient backgrounds**: Modern purple/pink theme
- **Color-coded risks**: Instant visual recognition
- **Smooth animations**: Professional feel
- **Responsive layout**: Works on all screen sizes
- **Status indicators**: Live chatbot status

---

## 🔧 Configuration Options

Edit `.env` file to customize:

### Model Parameters
```env
NVIDIA_TEMPERATURE=0.6    # Lower = more focused (0.0-1.0)
NVIDIA_TOP_P=0.95         # Nucleus sampling
NVIDIA_MAX_TOKENS=65536   # Max response length
```

### Risk Thresholds
```env
RISK_THRESHOLD_HIGH=0.7   # 70%+ = High Risk
RISK_THRESHOLD_MEDIUM=0.4 # 40%+ = Medium Risk
```

### Behavior Settings
```env
AUTO_PREDICT=true                  # Auto-predict on employee ID mention
CONTEXT_HISTORY_LENGTH=10          # Messages to keep in context
AUTO_SAVE_CONVERSATIONS=true       # Auto-save chats
```

---

## 📊 Advanced Features

### 1. Save Conversations
Click "💾 Save Conversation" to export your chat:
```json
{
  "session_id": "20251201_143022",
  "total_messages": 12,
  "prediction_count": 3,
  "conversation": [...]
}
```

### 2. View Statistics
Click "📊 Show Statistics" to see:
- Total messages in session
- Number of predictions made
- User vs assistant message count
- Data records loaded
- Model status

### 3. Quick Prompts
Use suggested prompts for common tasks:
- "What are the main causes of employee attrition?"
- "Analyze employee 150"
- "Show me high-risk employees"
- "What retention strategies work best?"
- "Explain the attrition model"

---

## 🐛 Troubleshooting

### Issue: "NVIDIA_API_KEY not found"
**Solution:**
1. Make sure you copied `.env.chatbot` to `.env`
2. Open `.env` and add your actual API key
3. OR set in PowerShell: `$env:NVIDIA_API_KEY="your_key"`
4. Restart the Streamlit app

### Issue: "Unable to import chatbot engine"
**Solution:**
```powershell
# Reinstall all dependencies
pip install -r requirements.txt

# If still failing, try upgrading pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: "Model not found"
**Solution:**
- Ensure `models/attrition_model.pkl` exists
- Check the MODEL_PATH in `.env` matches your file location
- The chatbot will still work without the model (no predictions)

### Issue: "Data not found"
**Solution:**
- Ensure `raw_data/employee_attrition_dataset.csv` exists
- Check the DATA_PATH in `.env` matches your file location
- The chatbot will still work (limited functionality)

### Issue: Slow responses
**Solution:**
1. Check your internet connection
2. NVIDIA API might be experiencing high load
3. Try reducing MAX_TOKENS in `.env`
4. Contact NVIDIA if issues persist

### Issue: Predictions not showing
**Solution:**
1. Mention employee ID explicitly: "analyze employee 150"
2. Check that ML model loaded successfully (see sidebar status)
3. Verify employee ID exists in dataset (1-1000 range)
4. Make sure AUTO_PREDICT=true in `.env`

---

## 🚀 Tips for Best Results

### Writing Effective Prompts
✅ **Good:**
- "Analyze employee 150 and suggest retention strategies"
- "What factors contribute most to attrition for employee 25?"
- "Compare the risk levels of employees 100, 200, and 300"

❌ **Less Effective:**
- "Employee" (too vague)
- "Risk" (no context)
- "Help" (unclear intent)

### Using Context
The chatbot remembers conversation history:
1. Ask about an employee
2. Follow up with "What should I do?" 
3. Request "Create an action plan"
4. The chatbot maintains context throughout

### Multiple Predictions
Analyze multiple employees in one conversation:
```
You: Analyze employee 100
[Chatbot shows prediction]

You: Now check employee 200
[Chatbot shows second prediction]

You: Which one needs more urgent attention?
[Chatbot compares both intelligently]
```

---

## 📈 What Makes This Chatbot Special?

### 1. **Fully Integrated ML Pipeline**
- Chatbot → Employee data → ML model → Prediction → AI insights
- All in one seamless conversation flow
- No manual steps required

### 2. **State-of-the-Art AI Model**
- NVIDIA Llama 3.3 Nemotron (49B parameters)
- Latest "Super" variant optimized for accuracy
- Professional-grade responses

### 3. **Real-Time Streaming**
- See responses as they're generated
- No waiting for complete answers
- Better user experience

### 4. **Persistent History**
- Every conversation saved
- Full context maintained
- Export to JSON for records

### 5. **Production-Ready UI**
- Professional gradient design
- Color-coded risk levels
- Intuitive navigation
- Mobile-responsive

---

## 🎓 Learning Resources

### Understanding the Technology

**NVIDIA Llama 3.3 Nemotron:**
- 49 billion parameters
- Trained on diverse datasets
- Optimized for instruction-following
- Supports up to 65K token responses

**ML Model:**
- XGBoost/CatBoost/Random Forest ensemble
- Trained on 1000+ employee records
- 24 input features
- Achieves 85%+ accuracy

**Integration:**
- Uses OpenAI-compatible API
- Streaming responses via SSE
- Context window management
- Automatic prompt engineering

### Further Reading
- [NVIDIA AI Platform Docs](https://docs.nvidia.com/ai/)
- [Llama Model Details](https://ai.nvidia.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Guide](https://platform.openai.com/docs/)

---

## 🤝 Support

### Getting Help
1. Check this QUICKSTART first
2. Review the `.env.chatbot` configuration
3. Check Streamlit console for error messages
4. Verify all files are in correct locations

### Common Questions

**Q: Is the NVIDIA API really free?**
A: Yes! NVIDIA provides free API access for development and testing.

**Q: Can I use a different AI model?**
A: Yes, modify the `model_name` in `nvidia_chatbot.py` or switch to OpenAI/Groq.

**Q: How many employees can it analyze?**
A: Unlimited! The dataset has 1000 employees, but the chatbot can handle any number.

**Q: Can I deploy this to production?**
A: Yes, but review NVIDIA's API terms for production use and rate limits.

**Q: Does it work offline?**
A: No, it requires internet connection for NVIDIA API calls.

---

## 📝 Next Steps

1. ✅ Get NVIDIA API key
2. ✅ Configure `.env` file
3. ✅ Install dependencies
4. ✅ Launch chatbot
5. 🎯 Try the example conversations above
6. 💡 Explore advanced features
7. 📊 Analyze your team's attrition risks
8. 🚀 Implement retention strategies

---

**Ready to start? Run:** `streamlit run app_chatbot.py`

**Questions? The chatbot itself can help!** Just ask: "How do I use this chatbot?"
