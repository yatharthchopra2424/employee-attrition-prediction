"""
Generative AI Engine Module
===========================
Provides LLM-powered recommendations, email drafts, and conversational AI
for the Retention Copilot.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

# Try to import LLM libraries
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RetentionAIEngine:
    """AI Engine for generating retention-focused content."""
    
    def __init__(self, api_key: str = None, provider: str = "groq"):
        self.provider = provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if provider == "groq" and GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)
            self.model = "llama-3.3-70b-versatile"
        elif provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = "gpt-4o-mini"
    
    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using the configured LLM."""
        if not self.client:
            return self._get_fallback_response(user_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._get_fallback_response(user_prompt)
    
    def _get_fallback_response(self, context: str) -> str:
        """Provide rule-based fallback when LLM is not available."""
        return "AI response generation requires API configuration. Please configure your Groq or OpenAI API key in Settings."


# Initialize global engine (will be configured later)
ai_engine = RetentionAIEngine()


def generate_retention_recommendation(
    employee_profile: Dict,
    risk_factors: List[Dict],
    risk_score: float
) -> str:
    """
    Generate personalized retention recommendations based on employee profile and risk factors.
    
    Args:
        employee_profile: Dictionary with employee information
        risk_factors: List of SHAP-derived risk factors
        risk_score: Overall attrition risk probability
    
    Returns:
        Markdown-formatted retention recommendations
    """
    system_prompt = """You are an expert HR consultant and retention strategist at Velorium Technologies. 
    Your role is to provide empathetic, actionable retention recommendations for managers.
    
    Guidelines:
    - Be specific and actionable
    - Maintain a warm, human-centered tone
    - Focus on the individual's unique situation
    - Prioritize immediate actions followed by long-term strategies
    - Always ground recommendations in the data provided
    - Remember: "Data didn't make us colder. It made us listen."
    """
    
    user_prompt = f"""Based on the following employee profile and risk analysis, provide personalized retention recommendations:

**Employee Profile:**
- Employee ID: {employee_profile.get('Employee_ID', 'N/A')}
- Department: {employee_profile.get('Department', 'N/A')}
- Role: {employee_profile.get('Job_Role', 'N/A')}
- Tenure: {employee_profile.get('Years_at_Company', 'N/A')} years
- Age: {employee_profile.get('Age', 'N/A')}
- Job Level: {employee_profile.get('Job_Level', 'N/A')}

**Risk Assessment:**
- Attrition Risk Score: {risk_score*100:.0f}%
- Risk Level: {'High' if risk_score >= 0.7 else 'Medium' if risk_score >= 0.4 else 'Low'}

**Top Risk Factors (from SHAP Analysis):**
{chr(10).join([f"- {f['factor']}: {f['value']:.2f} (Impact: {f['impact']})" for f in risk_factors[:5]])}

**Current Satisfaction Metrics:**
- Job Satisfaction: {employee_profile.get('Job_Satisfaction', 'N/A')}/5
- Work-Life Balance: {employee_profile.get('Work_Life_Balance', 'N/A')}/4
- Manager Relationship: {employee_profile.get('Relationship_with_Manager', 'N/A')}/5
- Years Since Last Promotion: {employee_profile.get('Years_Since_Last_Promotion', 'N/A')}

Please provide:
1. **Immediate Actions (This Week)** - 3 specific steps
2. **Short-Term Strategy (Next 30 Days)** - 3-4 initiatives
3. **Long-Term Retention Plan (Next Quarter)** - 2-3 strategic recommendations
4. **Key Conversation Points** - What the manager should discuss in their next 1:1
"""
    
    response = ai_engine._generate_response(system_prompt, user_prompt)
    
    if "API configuration" in response:
        # Provide structured fallback
        return _generate_fallback_recommendations(employee_profile, risk_factors, risk_score)
    
    return response


def _generate_fallback_recommendations(
    employee_profile: Dict,
    risk_factors: List[Dict],
    risk_score: float
) -> str:
    """Generate rule-based recommendations when LLM is not available."""
    
    recommendations = "## 🎯 Retention Recommendations\n\n"
    
    # Immediate Actions
    recommendations += "### Immediate Actions (This Week)\n\n"
    recommendations += "1. **Schedule a 1:1 Meeting** - Make it appreciative, not corrective. Acknowledge recent contributions.\n"
    
    # Check specific risk factors
    for factor in risk_factors:
        if 'Work_Life' in factor['factor'] and factor['impact'] == 'negative':
            recommendations += "2. **Discuss Workload** - Review current assignments and explore flexible arrangements.\n"
            break
    else:
        recommendations += "2. **Check-in on Wellbeing** - Have an open conversation about how they're feeling.\n"
    
    recommendations += "3. **Provide Recognition** - Publicly acknowledge a recent accomplishment.\n\n"
    
    # Short-term Strategy
    recommendations += "### Short-Term Strategy (Next 30 Days)\n\n"
    
    if employee_profile.get('Years_Since_Last_Promotion', 0) >= 3:
        recommendations += "1. **Career Path Discussion** - Create a visible growth trajectory with clear milestones.\n"
    else:
        recommendations += "1. **Development Planning** - Identify skill growth opportunities.\n"
    
    recommendations += "2. **Mentorship Connection** - Pair with a senior leader for guidance.\n"
    recommendations += "3. **Project Alignment** - Ensure assignments match career interests.\n\n"
    
    # Long-term Plan
    recommendations += "### Long-Term Retention Plan (Next Quarter)\n\n"
    recommendations += "1. **Promotion Evaluation** - Assess readiness for next level if performance warrants.\n"
    recommendations += "2. **Compensation Review** - Benchmark against market rates.\n"
    recommendations += "3. **Leadership Opportunities** - Assign ownership of a visible initiative.\n\n"
    
    return recommendations


def generate_email_draft(
    employee_profile: Dict,
    email_type: str = "appreciation",
    tone: str = "warm",
    context: str = ""
) -> str:
    """
    Generate an email draft for employee communication.
    
    Args:
        employee_profile: Dictionary with employee information
        email_type: Type of email (appreciation, growth_discussion, check_in, retention)
        tone: Email tone (formal, professional, warm, casual)
        context: Additional context for the email
    
    Returns:
        Email draft as string
    """
    system_prompt = f"""You are an expert at crafting empathetic, effective workplace communications.
    Write an email in a {tone} tone that feels genuine and human.
    
    Guidelines:
    - Be specific and personal, not generic
    - Lead with appreciation and positive intent
    - Keep it concise but meaningful
    - End with a clear call to action
    """
    
    email_types = {
        "appreciation": "Write an appreciation email recognizing the employee's contributions and value to the team.",
        "growth_discussion": "Write an email inviting the employee to discuss their career growth and development opportunities.",
        "check_in": "Write a check-in email to see how the employee is doing and offer support.",
        "retention": "Write an email expressing value for the employee and desire to support their continued success."
    }
    
    user_prompt = f"""{email_types.get(email_type, email_types['appreciation'])}

**Employee Context:**
- Role: {employee_profile.get('Job_Role', 'Team Member')}
- Department: {employee_profile.get('Department', 'Our Team')}
- Tenure: {employee_profile.get('Years_at_Company', 'several')} years
- Recent Performance: {employee_profile.get('Performance_Rating', 3)}/4

**Additional Context:**
{context if context else 'None provided'}

Generate a complete email with Subject line, greeting, body, and sign-off.
"""
    
    response = ai_engine._generate_response(system_prompt, user_prompt)
    
    if "API configuration" in response:
        return _generate_fallback_email(employee_profile, email_type, tone)
    
    return response


def _generate_fallback_email(employee_profile: Dict, email_type: str, tone: str) -> str:
    """Generate template-based email when LLM is not available."""
    
    role = employee_profile.get('Job_Role', 'Team Member')
    tenure = employee_profile.get('Years_at_Company', 'several')
    
    if email_type == "appreciation":
        return f"""**Subject:** Your Outstanding Contributions This Quarter

Hi [Employee Name],

I wanted to take a moment to personally acknowledge the exceptional work you've been delivering as our {role}. Your dedication and expertise over these {tenure} years have made a real difference to our team's success.

I recognize that the past few months have been demanding with tight deadlines and multiple project responsibilities. Your commitment during this period truly stands out, and I want you to know that it's deeply valued.

I'd love to connect with you this week to discuss your experience and explore ways we can better support your growth and wellbeing. Would you have 30 minutes for a conversation?

Thank you for being such an integral part of our team.

Warm regards,
[Manager Name]"""
    
    elif email_type == "growth_discussion":
        return f"""**Subject:** Let's Talk About Your Career Growth

Hi [Employee Name],

I've been thinking about your career development and would love to have a dedicated conversation about your aspirations and how we can better support your growth journey at Velorium.

You've built strong expertise in your role as {role} over the past {tenure} years, and I believe there are exciting opportunities ahead that align with your talents. I'd like to explore:

- Your career goals and where you'd like to be in 1-2 years
- Skills you're interested in developing
- Projects or responsibilities that would excite you
- Any certifications or training that would help your growth

Could we schedule 45 minutes this week? I'm looking forward to our discussion.

Best,
[Manager Name]"""
    
    else:
        return f"""**Subject:** Checking In - How Are You Doing?

Hi [Employee Name],

I hope you're doing well! I wanted to reach out and check in with you.

As someone who has been a valuable part of our team for {tenure} years, your wellbeing and satisfaction matter to me. I'd love to hear:

- How you're feeling about your current work
- Any challenges I can help with
- Ideas you have for improvement

Let me know if you'd like to grab a coffee and chat this week.

Best,
[Manager Name]"""


def generate_1on1_talking_points(
    employee_profile: Dict,
    risk_factors: List[Dict],
    focus_areas: List[str] = None
) -> str:
    """
    Generate talking points for a 1:1 meeting focused on retention.
    
    Args:
        employee_profile: Dictionary with employee information
        risk_factors: List of SHAP-derived risk factors
        focus_areas: Specific areas to focus on
    
    Returns:
        Structured talking points as string
    """
    system_prompt = """You are an expert coach helping managers have effective, empathetic 1:1 conversations.
    
    Guidelines:
    - Start with appreciation and positive intent
    - Use open-ended questions
    - Listen more than talk
    - Focus on understanding, not fixing
    - Make the employee feel valued and heard
    """
    
    focus = focus_areas or ["Career Growth", "Wellbeing", "Work Experience"]
    
    user_prompt = f"""Create a structured 1:1 conversation guide for a manager meeting with an at-risk employee.

**Employee Context:**
- Role: {employee_profile.get('Job_Role', 'Team Member')}
- Tenure: {employee_profile.get('Years_at_Company', 'N/A')} years
- Risk Score: {'High' if len(risk_factors) > 3 else 'Medium'}

**Key Risk Factors to Address:**
{chr(10).join([f"- {f['factor']}" for f in risk_factors[:4]])}

**Focus Areas:** {', '.join(focus)}

Provide:
1. Opening statements (2-3 appreciative conversation starters)
2. Discovery questions for each focus area
3. Commitment-building questions
4. Closing statements
"""
    
    response = ai_engine._generate_response(system_prompt, user_prompt)
    
    if "API configuration" in response:
        return _generate_fallback_talking_points(employee_profile, risk_factors)
    
    return response


def _generate_fallback_talking_points(employee_profile: Dict, risk_factors: List[Dict]) -> str:
    """Generate template-based talking points when LLM is not available."""
    
    return f"""## 1:1 Conversation Guide

### 🎯 Opening (Set Positive Tone)
- "I wanted to take some time to check in with you and acknowledge the incredible work you've been putting in."
- "Your contributions haven't gone unnoticed, and I value having you on the team."
- "This is a safe space - I genuinely want to understand how you're doing."

### 💡 Discovery Questions

**On Workload & Balance:**
- "How are you feeling about your current workload?"
- "Is there anything we can adjust to help you maintain a healthier balance?"
- "What would make your day-to-day work more manageable?"

**On Career Growth:**
- "Where do you see yourself in the next year or two?"
- "What skills are you most excited about developing?"
- "Are there projects or opportunities you'd like to explore?"

**On Support & Resources:**
- "What support do you need from me that you're not getting?"
- "Are there any obstacles I can help remove?"
- "How can I better support your success?"

### 🤝 Commitment Building
- "Based on our conversation, here's what I commit to doing..."
- "Let's set up a follow-up in two weeks to check progress."
- "Your growth and wellbeing matter to me and to the team."

### ✨ Closing
- "Thank you for being open with me today."
- "I'm committed to making sure you feel valued and supported here."
- "Let's stay connected - my door is always open."
"""


def generate_policy_suggestion(
    department: str,
    risk_factors: List[str],
    employee_count: int
) -> str:
    """
    Generate policy suggestions based on department-wide risk patterns.
    
    Args:
        department: Department name
        risk_factors: Common risk factors in the department
        employee_count: Number of at-risk employees
    
    Returns:
        Policy suggestion document
    """
    system_prompt = """You are an HR policy expert helping organizations develop retention-focused policies.
    
    Guidelines:
    - Be practical and implementable
    - Consider cost-effectiveness
    - Focus on systemic improvements
    - Balance employee needs with business requirements
    """
    
    user_prompt = f"""Based on the following department analysis, suggest policy improvements:

**Department:** {department}
**At-Risk Employees:** {employee_count}
**Common Risk Factors:**
{chr(10).join([f"- {f}" for f in risk_factors])}

Provide:
1. Top 3 policy recommendations with implementation details
2. Expected impact on retention
3. Resource requirements
4. Timeline for implementation
"""
    
    response = ai_engine._generate_response(system_prompt, user_prompt)
    
    if "API configuration" in response:
        return f"""## Policy Recommendations for {department}

### 1. Flexible Work Arrangements
- Implement hybrid work options
- Allow flexible hours where possible
- Expected Impact: 15-20% improvement in work-life balance scores

### 2. Career Development Program
- Quarterly career conversations
- Clear promotion criteria and timelines
- Skills development budget per employee
- Expected Impact: Reduced promotion-related attrition

### 3. Manager Training Initiative
- Training on retention conversations
- Recognition best practices
- Early warning sign identification
- Expected Impact: Improved manager-employee relationships

### Implementation Timeline: 3-6 months
### Resource Requirement: HR team + Department leadership alignment
"""
    
    return response


def chat_with_copilot(
    user_message: str,
    conversation_history: List[Dict],
    employee_context: Dict = None
) -> str:
    """
    Handle conversational interactions with the Retention Copilot.
    
    Args:
        user_message: User's message
        conversation_history: Previous messages in the conversation
        employee_context: Optional context about employees being discussed
    
    Returns:
        AI response as string
    """
    system_prompt = """You are the Velorium Retention Copilot, an AI assistant helping managers retain valuable employees.

Your capabilities:
- Analyze employee attrition risk
- Provide personalized retention recommendations
- Generate communication drafts
- Answer questions about retention strategies
- Provide data-driven insights

Your personality:
- Empathetic and supportive
- Data-driven but human-centered
- Practical and actionable
- Always remember: "Data didn't make us colder. It made us listen."

When you don't have specific employee data, provide general best practices and offer to help with specific cases.
"""
    
    # Build conversation context
    context_str = ""
    if employee_context:
        context_str = f"\n\nCurrent Employee Context:\n{str(employee_context)}"
    
    history_str = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in conversation_history[-6:]  # Last 6 messages for context
    ])
    
    user_prompt = f"""Conversation History:
{history_str}

{context_str}

User: {user_message}

Provide a helpful, actionable response. If the user asks about specific employees or data you don't have, offer to help with the information they can provide or suggest using the Employee Analysis feature."""
    
    response = ai_engine._generate_response(system_prompt, user_prompt)
    
    if "API configuration" in response:
        return _generate_fallback_chat_response(user_message)
    
    return response


def _generate_fallback_chat_response(user_message: str) -> str:
    """Generate a helpful response when LLM is not available."""
    
    user_lower = user_message.lower()
    
    if "high risk" in user_lower or "at risk" in user_lower:
        return """Based on our analysis, high-risk employees typically share these characteristics:

1. **Work-Life Balance Issues** - Working excessive hours, poor balance ratings
2. **Career Stagnation** - No promotion in 3+ years
3. **Engagement Decline** - Dropping satisfaction scores
4. **Manager Relationship** - Below average relationship quality

**Recommended Actions:**
- Use the Employee Analysis feature to view specific at-risk employees
- Schedule proactive 1:1s focused on listening, not correcting
- Review workload distribution in affected teams

Would you like me to help you:
- Analyze a specific employee?
- Generate retention recommendations?
- Draft a communication?"""
    
    elif "email" in user_lower or "draft" in user_lower:
        return """I can help you draft various types of communications:

1. **Appreciation Email** - Recognize contributions
2. **Growth Discussion Invite** - Career conversation starter
3. **Check-in Email** - Wellbeing-focused outreach
4. **Retention Conversation** - For at-risk employees

To generate a personalized draft, please:
1. Go to the **Action Generator** section
2. Select the communication type
3. Choose the employee
4. Customize the tone

Would you like guidance on any specific type of communication?"""
    
    elif "retention" in user_lower or "strategy" in user_lower:
        return """Effective retention strategies based on our data analysis:

**Immediate Impact:**
- Regular appreciation and recognition
- Workload balance reviews
- Flexible work arrangements

**Medium-Term:**
- Clear career progression paths
- Skills development opportunities
- Mentorship programs

**Long-Term:**
- Competitive compensation reviews
- Leadership development
- Culture of psychological safety

**Key Insight:** Strong manager relationships can offset up to 40% of compensation-related attrition risk.

Would you like specific recommendations for a department or employee?"""
    
    else:
        return """I'm here to help you retain your valuable employees! Here's what I can assist with:

📊 **Analysis**
- Employee attrition risk assessment
- Department-wide risk patterns
- Key risk factor identification

💡 **Recommendations**
- Personalized retention strategies
- 1:1 conversation guides
- Policy suggestions

📧 **Communications**
- Appreciation emails
- Career discussion invites
- Check-in messages

🔍 **Insights**
- SHAP-based risk factor explanations
- Engagement metric analysis
- Historical pattern recognition

How can I help you today?"""


def configure_ai_engine(api_key: str, provider: str = "groq"):
    """
    Configure the AI engine with API credentials.
    
    Args:
        api_key: API key for the chosen provider
        provider: 'groq' or 'openai'
    """
    global ai_engine
    ai_engine = RetentionAIEngine(api_key=api_key, provider=provider)
