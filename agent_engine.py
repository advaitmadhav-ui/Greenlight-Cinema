import json
import re
import sys
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Define the Memory State
class AgentState(TypedDict):
    prompt: str
    draft: str
    feedback: str
    revision_count: int
    status: str
    cast: str
    score: Optional[str]
    quarter: Optional[str]

# 2. Initialize LLM, Local Services, and Data
print("🧠 Booting up Qwen2.5:1.5b via Ollama...")
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)

print("🗄️ Connecting to local ChromaDB using Nomic embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Safely load market constraints
try:
    with open("market_constraints.json", "r", encoding="utf-8") as f:
        market_constraints = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print("⚠️ Warning: market_constraints.json not found or invalid. Using defaults.")
    market_constraints = {"notice": "No market constraints found. Evaluate based on general blockbuster appeal."}

# Safely load face constraints
try:
    with open("face.constraint", "r", encoding="utf-8") as f:
        face_constraints = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print("⚠️ Warning: face.constraint not found or invalid. Using empty casting roster.")
    face_constraints = {}

# 3. Define the Writer Agent
def writer_node(state: AgentState):
    print(f"\n✍️ WRITER AGENT: Brainstorming and drafting...")
    
    # Safely attempt to fetch from ChromaDB
    try:
        docs = vectordb.similarity_search(state['prompt'], k=2)
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"⚠️ Warning: Could not search ChromaDB ({e}). Proceeding without context.")
        context = "No additional context provided."
    
    sys_msg = (
        "You are an elite, A-list Hollywood screenwriter and pitch master, renowned for crafting high-concept, highly profitable blockbusters. "
        "Your task is to write a visceral, cinematic, and compelling 2-paragraph movie pitch based on the user's prompt.\n\n"
        "STRUCTURE REQUIREMENTS:\n"
        "- Paragraph 1 (The Hook & Setup): Establish the world, the protagonist, the inciting incident, and the core conflict.\n"
        "- Paragraph 2 (The Escalation & Stakes): Detail the rising action, the primary threat/antagonist, and the high-stakes climax.\n\n"
        "STYLE & TONE:\n"
        "- Make it punchy, visual, and fast-paced. Use active voice and strong verbs.\n"
        "- Absorb the style and atmosphere from the provided 'Inspiration Context', but DO NOT copy it directly.\n\n"
        "STRICT RULES:\n"
        "- Output EXACTLY two paragraphs. No more, no less.\n"
        "- The scale of the story MUST respect any budget constraints mentioned in the prompt (e.g., a $20M budget dictates contained locations and psychological tension, not sprawling CGI armies).\n"
        "- Output ONLY the pitch. No introductory greetings, titles, or meta-commentary."
    )
    user_prompt = f"Prompt: {state['prompt']}\n\nInspiration Context:\n{context}"
    
    response = llm.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=user_prompt)
    ])
    
    return {"draft": response.content, "revision_count": state['revision_count']}

# 4. Define the Critic Agent
def critic_node(state: AgentState):
    print(f"\n🧐 CRITIC AGENT: Evaluating draft against market data (Revision {state['revision_count']})...")
    
    if state['revision_count'] >= 3:
        print("⚠️ Max revisions reached. Forcing ACCEPT to avoid infinite loop.")
        return {
            "feedback": "Max revisions reached. Passing as is.",
            "status": "ACCEPT",
            "score": None,
            "quarter": None,
        }
        
    sys_msg = (
        "You are a ruthless Hollywood studio executive focused entirely on ROI. "
        "Evaluate the script draft against the provided market constraints. "
        "You must decide whether to 'ACCEPT' or 'REJECT' the draft.\n\n"
        "RULES:\n"
        "- If it clearly aligns with the constraints (e.g., uses high-ROI genres or seasonal fits), start your response EXACTLY with 'STATUS: ACCEPT'.\n"
        "- If it fails, start your response EXACTLY with 'STATUS: REJECT' and provide 2-3 sentences of harsh, specific feedback on what to change.\n\n"
        "You recommend which Quarter the movie must be realised according to choose genre. "
        "State this exactly as 'Quarter: Q1' (or Q2/Q3/Q4) on its own line.\n"
        "And also give the script a score when you 'ACCEPT' or Max revision are reached. "
        "State this exactly as 'Score: --/10' on its own line.\n"
        "And only show the final draft\n"
        f"Market Constraints:\n{json.dumps(market_constraints, indent=2)}"
    )
    
    response = llm.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=f"Current Draft:\n{state['draft']}")
    ])
    
    content = response.content
    
    if "STATUS: ACCEPT" in content.upper():
        status = "ACCEPT"
        print("✅ CRITIC DECISION: ACCEPTED!")
    else:
        status = "REJECT"
        print(f"❌ CRITIC DECISION: REJECTED! Feedback: {content}")

    # Extract the score and quarter the critic was asked to provide, instead
    # of letting them get discarded inside the raw feedback text blob.
    score_match = re.search(r"score\s*:\s*([\d.]+\s*/\s*10)", content, re.IGNORECASE)
    quarter_match = re.search(r"quarter\s*:\s*(Q[1-4])", content, re.IGNORECASE)
    score = score_match.group(1).replace(" ", "") if score_match else None
    quarter = quarter_match.group(1).upper() if quarter_match else None

    return {"feedback": content, "status": status, "score": score, "quarter": quarter}

# 5. Define the Refiner Agent
def refiner_node(state: AgentState):
    print("\n🛠️ REFINER AGENT: Incorporating feedback and rewriting...")
    
    sys_msg = (
        "You are a highly adaptable script doctor. "
        "Rewrite the provided draft to perfectly address the studio executive's feedback. "
        "Do not apologize or explain, just output the newly revised 2-paragraph synopsis."
    )
    user_prompt = f"Original Draft:\n{state['draft']}\n\nExecutive Feedback:\n{state['feedback']}\n\nRewrite the synopsis:"
    
    response = llm.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=user_prompt)
    ])
    
    return {"draft": response.content, "revision_count": state['revision_count'] + 1}

# 6. Define the Casting Agent
def casting_node(state: AgentState):
    print("\n🎭 CASTING AGENT: Sourcing high-ROI talent for the approved script...")
    
    sys_msg = (
        "You are a strategic Hollywood Casting Director. "
        "Based on the approved script draft, the budget mentioned in the prompt, and the market constraints provided, "
        "recommend 2 to 3 specific actors from the constraints list to lead the film.\n\n"
        "CASTING RULES:\n"
        "- If the budget mentioned in the prompt is under $40 Million, prioritize actors from the 'Emerging Actors' list to keep costs down while maximizing ROI.\n"
        "- If the budget is $40 Million or over, prioritize actors from the 'Top Actors' (Mainstream) list to guarantee box office draw.\n"
        "- Briefly explain why you chose them based on the script's genre and their ROI.\n\n"
        "For each actor you recommend, you MUST include their image URL from this list: "
        f"{json.dumps(face_constraints, indent=2)} "
        "Format your output so the actor name is followed by their URL in brackets, e.g.: [Name](URL)\n"
        f"Market Constraints:\n{json.dumps(market_constraints, indent=2)}"
    )
    user_prompt = f"Original Prompt (Contains Budget): {state['prompt']}\n\nApproved Draft:\n{state['draft']}\n\nProvide your casting recommendations:"
    
    response = llm.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=user_prompt)
    ])
    
    return {"cast": response.content}

# 7. Define the Routing Logic
def router(state: AgentState):
    if state["status"] == "ACCEPT":
        return "casting" # Route to casting instead of END
    return "refiner"

# 8. Build and Wire the Graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("casting", casting_node)
    
    workflow.set_entry_point("writer")
    
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges(
        "critic",
        router,
        {
            "refiner": "refiner",  
            "casting": "casting"   
        }
    )
    workflow.add_edge("refiner", "critic") 
    workflow.add_edge("casting", END)
    
    return workflow.compile()


def quick_revise(current_draft: str, user_request: str) -> str:
    print("\n⚡ QUICK REVISION: Bypassing graph for direct edit...")
    
    sys_msg = (
        "You are an expert Hollywood script doctor. "
        "The user wants to make a change to an existing movie pitch. "
        "Revise the text according to their instructions. "
        "The provided text may contain formatting. Output the revised version cleanly in Markdown. "
        "Do not include meta-commentary like 'Here is the revised version'."
    )
    user_msg = f"CURRENT PITCH:\n{current_draft}\n\nUSER INSTRUCTIONS:\n{user_request}\n\nREVISED PITCH:"
    
    response = llm.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=user_msg)
    ])
    
    return response.content

# =====================================================================

if __name__ == "__main__":
    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Write a Horror movie pitch. The production budget is capped at $20 Million. Core idea: A terrifying monster stalks a crew on a deep sea oil rig."
    
    print(f"\n🚀 INITIALIZING ENGINE WITH PROMPT: '{user_input}'\n")
    engine = build_graph()
    
    initial_state = {
        "prompt": user_input,
        "draft": "", 
        "feedback": "None yet", 
        "revision_count": 0, 
        "status": "PENDING",
        "cast": "",
        "score": None,
        "quarter": None,
    }
    
    final_state = engine.invoke(initial_state)
    
    print("\n" + "="*80)
    print("🎬 FINAL APPROVED SCRIPT PITCH:")
    print("="*80)
    print(final_state["draft"])
    print("\n" + "="*80)
    print("🎭 CASTING RECOMMENDATIONS:")
    print("="*80)
    print(final_state["cast"])
    print("="*80)
    print(f"📊 Score: {final_state.get('score') or 'N/A'}  |  Recommended Quarter: {final_state.get('quarter') or 'N/A'}")
    print("="*80 + "\n")