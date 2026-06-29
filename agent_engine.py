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
    title: str
    draft: str
    feedback: str
    revision_count: int
    status: str
    cast: str
    cast_list: list
    score: Optional[str]
    quarter: Optional[str]

# 2. Initialize LLM, Local Services, and Data
# Separate LLM "personas" per agent role — same model, tuned generation params only.
# CPU-bound inference: every saved output token is real wall-clock time, so
# num_predict caps are set tight to each agent's actual job. No graph/routing
# logic is touched anywhere below.
print("🧠 Booting up Qwen2.5:1.5b via Ollama...")
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)  # kept as-is for any external import of `llm`

# Writer: creative prose — needs some temperature headroom, capped length so it
# doesn't run past 2 paragraphs worth of tokens.
llm_writer = ChatOllama(model="qwen2.5:1.5b", temperature=0.65, top_p=0.9, num_predict=480)

# Critic: evaluative/format-following task. Lower temperature makes it far more
# likely to nail "STATUS: ACCEPT/REJECT", "Score:", "Quarter:" on the first try
# (fewer malformed outputs for the regex parser) — and lower temperature costs
# nothing in latency. Short cap since feedback only needs 2-3 sentences.
llm_critic = ChatOllama(model="qwen2.5:1.5b", temperature=0.1, top_p=0.85, num_predict=300)

# Refiner: rewriting task — some creativity needed but should stay anchored
# to the original draft, so a middle-ground temperature.
llm_refiner = ChatOllama(model="qwen2.5:1.5b", temperature=0.5, top_p=0.9, num_predict=420)

# Casting: pure selection from a fixed roster, not generation — should be
# near-deterministic and only needs a couple of short lines.
llm_casting = ChatOllama(model="qwen2.5:1.5b", temperature=0.25, top_p=0.85, num_predict=200)

# Intent router (used in quick_revise): tiny classification output, needs to
# be deterministic and fast.
llm_router = ChatOllama(model="qwen2.5:1.5b", temperature=0.1, num_predict=30)

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
        "OUTPUT FORMAT (follow exactly):\n"
        "Title: <a punchy, marketable movie title, no quotes or extra punctuation around it>\n"
        "<blank line>\n"
        "<paragraph 1>\n\n"
        "<paragraph 2>\n\n"
        "STRUCTURE REQUIREMENTS:\n"
        "- Paragraph 1 (The Hook & Setup): Establish the world, the protagonist, the inciting incident, and the core conflict.\n"
        "- Paragraph 2 (The Escalation & Stakes): Detail the rising action, the primary threat/antagonist, and the high-stakes climax.\n\n"
        "STYLE & TONE:\n"
        "- Make it punchy, visual, and fast-paced. Use active voice and strong verbs.\n"
        "- Absorb the style and atmosphere from the provided 'Inspiration Context', but DO NOT copy it directly.\n\n"
        "STRICT RULES:\n"
        "- The FIRST line of your output MUST be exactly 'Title: <movie title>' and nothing else on that line.\n"
        "- After the title line, output EXACTLY two paragraphs of pitch. No more, no less.\n"
        "- The scale of the story MUST respect any budget constraints mentioned in the prompt (e.g., a $20M budget dictates contained locations and psychological tension, not sprawling CGI armies).\n"
        "- Output ONLY the title line followed by the pitch. No introductory greetings, headers, or meta-commentary."
    )
    user_prompt = f"Prompt: {state['prompt']}\n\nInspiration Context:\n{context}"
    
    response = llm_writer.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=user_prompt)
    ])
    
    raw = response.content

    # Primary: look for the instructed "Title: ..." line, tolerating stray
    # markdown bolding/asterisks around it (e.g. "**Title:** X" or "*Title*: X").
    title_match = re.search(
        r"^\s*\**\s*Title\s*\**\s*:\s*\**\s*(.+?)\s*\**\s*$",
        raw, re.IGNORECASE | re.MULTILINE
    )
    if title_match:
        movie_title = title_match.group(1).strip().strip('"').strip("'")
        draft = (raw[:title_match.start()] + raw[title_match.end():]).strip()
    else:
        # Fallback: the model sometimes skips the "Title:" line and instead
        # names the movie inline, e.g. '**The Ransom Race** promises...' or
        # 'titled "The Ransom Race",'. Try to recover it before giving up.
        movie_title = None
        inline_patterns = [
            r'\*\*([A-Z][^*]{2,60})\*\*\s+(?:promises|is|tells|follows|delivers|brings)',
            r'(?:titled|called)\s+["\u201c]([^"\u201d]{2,60})["\u201d]',
            r'^["\u201c]([^"\u201d]{2,60})["\u201d]',
        ]
        for pattern in inline_patterns:
            inline_match = re.search(pattern, raw, re.IGNORECASE | re.MULTILINE)
            if inline_match:
                movie_title = inline_match.group(1).strip()
                break
        draft = raw.strip()
        if not movie_title:
            movie_title = "Untitled Pitch"

    return {"title": movie_title, "draft": draft, "revision_count": state['revision_count']}

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
        "- If it aligns with the constraints (e.g., uses high-ROI genres), start your response EXACTLY with 'STATUS: ACCEPT'.\n"
        "- If it fails, start your response EXACTLY with 'STATUS: REJECT'.\n\n"
        "After the STATUS line, give your reasoning as 6-7 short bullet points (one line each, "
        "starting with '- '). Each bullet should be a single sharp, specific observation — "
        "covering things like genre/ROI fit, budget alignment, pacing, character, marketability, "
        "and (if rejecting) the single most important fix needed. No long paragraphs, no intro "
        "sentence before the bullets — go straight from the STATUS line into the bullet list.\n\n"
        "You recommend which Quarter the movie must be realised according to choose genre. "
        "State this exactly as 'Quarter: Q1' (or Q2/Q3/Q4) on its own line.\n"
        "And also give the script a score when you 'ACCEPT' or Max revision are reached. "
        "State this exactly as 'Score: --/10' on its own line.\n"
        f"Market Constraints:\n{json.dumps(market_constraints, indent=2)}"
    )
    
    response = llm_critic.invoke([
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

    score_match = re.search(r"score\s*:\s*([\d.]+\s*/\s*10)", content, re.IGNORECASE)
    quarter_match = re.search(r"quarter\s*:\s*(Q[1-4])", content, re.IGNORECASE)
    score = score_match.group(1).replace(" ", "") if score_match else None
    quarter = quarter_match.group(1).upper() if quarter_match else None

    return {"feedback": content, "status": status, "score": score, "quarter": quarter}

# 5. Define the Refiner Agent
def refiner_node(state: AgentState):
    print("\n🛠️ REFINER AGENT: Incorporating feedback and rewriting...")
    
    # state['feedback'] is the critic's raw response, which includes the literal
    # "STATUS: REJECT" line (and sometimes Score:/Quarter: lines). Feeding that
    # verbatim primes a small model to keep talking *like* a critic instead of
    # writing a pitch. Strip those control lines, keep only the actual prose feedback.
    feedback_lines = state['feedback'].splitlines()
    clean_feedback = "\n".join(
        line for line in feedback_lines
        if not re.match(r"^\s*(STATUS|SCORE|QUARTER)\s*:", line, re.IGNORECASE)
    ).strip()
    if not clean_feedback:
        clean_feedback = "Tighten pacing and strengthen the central conflict."

    sys_msg = (
        "You are a highly adaptable script doctor. "
        "Rewrite the provided movie pitch draft to address the feedback below.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY the revised pitch itself: exactly 2 paragraphs of cinematic prose.\n"
        "- Do NOT mention budgets, ROI, market constraints, scores, or studio decisions as commentary — "
        "if the feedback raises one of these, address it BY CHANGING THE STORY, not by writing about it.\n"
        "- Do NOT include words like 'STATUS', 'REJECT', 'ACCEPT', 'feedback', or any meta-commentary.\n"
        "- Do not apologize or explain. Output the pitch prose and nothing else."
    )
    user_prompt = (
        f"Original Draft:\n{state['draft']}\n\n"
        f"Feedback to address (rewrite the STORY to fix this, do not discuss it):\n{clean_feedback}\n\n"
        "Revised 2-paragraph pitch:"
    )
    
    response = llm_refiner.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=user_prompt)
    ])
    
    return {"draft": response.content, "revision_count": state['revision_count'] + 1}

# 6. Define the Casting Agent
def _extract_budget_millions(prompt_text: str):
    patterns = [
        r"\$\s*([\d,.]+)\s*(?:million|mil|m)\b",
        r"\b([\d,.]+)\s*million\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None

def _build_candidate_roster(tier_list, face_constraints, limit=12):
    candidates = []
    for entry in tier_list:
        name_match = re.match(r"^(.*?)\s*\(", entry)
        name = name_match.group(1).strip() if name_match else entry.strip()
        url = face_constraints.get(name)
        if url:
            candidates.append({"name": name, "roi_label": entry, "url": url})
    return candidates[:limit]

def casting_node(state: AgentState):
    print("\n🎭 CASTING AGENT: Sourcing high-ROI talent for the approved script...")

    budget = _extract_budget_millions(state['prompt'])
    low_budget = budget is not None and budget < 40

    emerging = market_constraints.get("emerging_actors", [])
    mainstream = market_constraints.get("top_actors", [])
    tier_list = emerging if low_budget else mainstream
    tier_label = "Emerging Actors" if low_budget else "Top Actors (Mainstream)"

    roster = _build_candidate_roster(tier_list, face_constraints)

    if not roster:
        fallback_list = mainstream if low_budget else emerging
        roster = _build_candidate_roster(fallback_list, face_constraints)
        if roster:
            tier_label = "Top Actors (Mainstream)" if low_budget else "Emerging Actors"

    if not roster:
        msg = "No casting data is currently available — check that market_constraints.json and face.constraint are present and contain matching actor names."
        return {"cast": msg, "cast_list": []}

    roster_by_name = {c['name'].lower(): c for c in roster}
    roster_block = "\n".join(f"- {c['name']} — {c['roi_label']}" for c in roster)

    budget_note = f"${budget:.0f} Million" if budget is not None else "unspecified (no explicit figure found in the prompt)"

    sys_msg = (
        "You are a strategic Hollywood Casting Director. "
        "Recommend 2 to 3 actors to lead the film, choosing ONLY from the candidate roster below — "
        "do not suggest anyone who is not on this list.\n\n"
        f"CANDIDATE ROSTER ({tier_label}, budget tier already selected for you — budget: {budget_note}):\n"
        f"{roster_block}\n\n"
        "OUTPUT FORMAT (follow exactly, one actor per line, no numbering, no markdown, no extra symbols):\n"
        "Name | One-sentence reasoning referencing their ROI figure and fit for the genre.\n\n"
        "INSTRUCTIONS:\n"
        "- Pick 2-3 names from the roster above, spelled exactly as shown.\n"
        "- Output ONLY the 'Name | reasoning' lines, nothing before or after them.\n"
    )
    user_prompt = f"Approved Draft:\n{state['draft']}\n\nProvide your casting recommendations from the roster:"

    response = llm_casting.invoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_prompt)
    ])

    cast_list = []
    for line in response.content.splitlines():
        line = line.strip().lstrip("-•*0123456789. ").strip()
        if "|" not in line:
            continue
        name_part, _, reasoning_part = line.partition("|")
        name_part = name_part.strip()
        candidate = roster_by_name.get(name_part.lower())
        if candidate:
            cast_list.append({
                "name": candidate["name"],
                "roi_label": candidate["roi_label"],
                "url": candidate["url"],
                "reasoning": reasoning_part.strip(),
            })

    if not cast_list:
        for c in roster[:3]:
            cast_list.append({
                "name": c["name"],
                "roi_label": c["roi_label"],
                "url": c["url"],
                "reasoning": "Selected from the verified casting roster for this budget tier.",
            })


    cast_text = "\n".join(
        f"* **{c['name']}** ({c['roi_label'].split('(')[-1].rstrip(')')}): {c['reasoning']}"
        if "(" in c['roi_label'] else f"* **{c['name']}**: {c['reasoning']}"
        for c in cast_list
    )

    return {"cast": cast_text, "cast_list": cast_list}


# 7. Define the Routing Logic
def router(state: AgentState):
    if state["status"] == "ACCEPT":
        return "casting" 
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


# =====================================================================
# THE UPGRADED SUPERVISOR REFINER
# =====================================================================
def quick_revise(current_state: AgentState, user_request: str) -> AgentState:
    print("\n⚡ SUPERVISOR REFINER: Analyzing request to route to specialized agents...")
    
    # 1. Intent Routing: Ask the LLM which departments to activate
    sys_msg = (
        "You are a Studio Executive routing a requested script change to the correct departments.\n"
        "Analyze the user's request and respond ONLY with a comma-separated list of these exact tags:\n"
        "- SCRIPT (if the story, plot, title, or budget needs changing)\n"
        "- CRITIC (if the user explicitly asks for a new market evaluation or score)\n"
        "- CASTING (if the user wants different actors or a recast)\n\n"
        "Example: SCRIPT, CASTING"
    )
    
    response = llm_router.invoke([
        SystemMessage(content=sys_msg), 
        HumanMessage(content=f"User Request: {user_request}")
    ])
    
    intent = response.content.upper()
    print(f"🔀 ROUTER DECISION: Activated Departments -> {intent}")
    
    # Clone the state and append the new request so downstream agents (like casting) 
    # can catch new budget constraints if the user mentioned them.
    new_state = current_state.copy()
    new_state['prompt'] = new_state['prompt'] + f" \nRevision Request: {user_request}"
    
    # 2. Targeted Execution: Run only the requested agents
    if "SCRIPT" in intent:
        print("✍️ REFINER: Editing the draft...")
        refine_sys = (
            "You are an expert Hollywood script doctor. "
            "Revise the pitch to concretely reflect the user's instruction below.\n\n"
            "IMPORTANT: if the instruction is a fact or number (e.g. a new budget figure, "
            "a renamed character, a different setting) rather than a narrative note, you must "
            "still produce a VISIBLY DIFFERENT rewrite that reflects it in the actual prose — "
            "do not just silently accept the fact without changing anything. For a budget change "
            "specifically: a bigger budget should read as more spectacle, more locations, larger "
            "set-pieces; a smaller budget should read as more contained, fewer locations, tighter "
            "psychological tension. Always rewrite scale/scope language to match.\n\n"
            "Do not include meta-commentary. Output the revised 2-paragraph pitch cleanly, "
            "and nothing else."
        )
        refine_msg = (
            f"CURRENT PITCH:\n{new_state['draft']}\n\n"
            f"INSTRUCTION TO APPLY:\n{user_request}\n\n"
            "Rewrite the pitch now so this instruction is concretely visible in the prose:"
        )
        res = llm_refiner.invoke([SystemMessage(content=refine_sys), HumanMessage(content=refine_msg)])
        new_state['draft'] = res.content.strip()

    if "CRITIC" in intent:
        # Reset the revision count so the critic actually evaluates it
        new_state['revision_count'] = 0 
        # UPDATE the dictionary instead of overwriting it
        new_state.update(critic_node(new_state))

    if "CASTING" in intent:
        # UPDATE the dictionary instead of overwriting it
        new_state.update(casting_node(new_state))
        
    return new_state

# =====================================================================

if __name__ == "__main__":
    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Write a Horror movie pitch. The production budget is capped at $20 Million. Core idea: A terrifying monster stalks a crew on a deep sea oil rig."
    
    print(f"\n🚀 INITIALIZING ENGINE WITH PROMPT: '{user_input}'\n")
    engine = build_graph()
    
    initial_state = {
        "prompt": user_input,
        "title": "",
        "draft": "", 
        "feedback": "None yet", 
        "revision_count": 0, 
        "status": "PENDING",
        "cast": "",
        "cast_list": [],
        "score": None,
        "quarter": None,
    }
    
    final_state = engine.invoke(initial_state)
    
    print("\n" + "="*80)
    print(f"🎬 FINAL APPROVED SCRIPT PITCH: {final_state.get('title', 'Untitled')}")
    print("="*80)
    print(final_state["draft"])
    print("\n" + "="*80)
    print("🎭 CASTING RECOMMENDATIONS:")
    print("="*80)
    print(final_state["cast"])
    print("="*80)
    print(f"📊 Score: {final_state.get('score') or 'N/A'}  |  Recommended Quarter: {final_state.get('quarter') or 'N/A'}")
    print("="*80 + "\n")