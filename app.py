"""
A tiny LangGraph app: a support-ticket router.

Flow:  START -> classify -> (billing | tech | shipping | general) -> review --(retry?)--> END
                                                            |
                                                            +--> back to the agent (a cycle)

Concepts it shows:
  1. State      - a TypedDict shared by every node
  2. Nodes      - plain functions that take state and return a partial update
  3. Edges      - fixed edges and conditional (branching) edges
  4. Reducers   - how an update is merged into state (here: append to a list)
  5. Cycles     - a node can route back to an earlier one (LangGraph's big win over chains)
  6. Memory     - a checkpointer saves state per thread_id between calls

The agents answer with Claude (via langchain-anthropic) when ANTHROPIC_API_KEY is set;
otherwise they fall back to canned keyword-stub answers so the demo still runs offline.
"""

import operator
import os
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


# 1. STATE -----------------------------------------------------------------
class State(TypedDict):
    question: str
    category: str
    answer: str
    attempts: int
    # Reducer: updates to `log` are APPENDED instead of overwriting.
    log: Annotated[list[str], operator.add]


# 2. NODES -----------------------------------------------------------------
llm = None
if os.environ.get("ANTHROPIC_API_KEY"):
    from langchain_anthropic import ChatAnthropic

    # effort="low" keeps short support replies fast/cheap; Opus 5.5 always thinks adaptively.
    llm = ChatAnthropic(model="claude-opus-5-5", max_tokens=4096, effort="low")


def ask_claude(role: str, question: str, fallback: str) -> str:
    if llm is None:
        return fallback
    msg = llm.invoke([
        ("system", f"You are a {role} support agent. Reply in 1-3 sentences."),
        ("user", question),
    ])
    return msg.text


def classify(state: State) -> dict:
    q = state["question"].lower()
    if any(w in q for w in ("refund", "charge", "invoice", "bill")):
        cat = "billing"
    elif any(w in q for w in ("error", "crash", "bug", "login")):
        cat = "tech"
    elif any(w in q for w in ("delivery", "package", "shipping", "tracking")):
        cat = "shipping"
    else:
        cat = "general"
    return {"category": cat, "attempts": 0, "log": [f"classify -> {cat}"]}


def billing_agent(state: State) -> dict:
    ans = ask_claude("billing", state["question"], "I've opened a refund request for you.")
    return {"answer": ans,
            "attempts": state["attempts"] + 1, "log": ["billing_agent answered"]}


def tech_agent(state: State) -> dict:
    # First attempt is deliberately weak so we can see the retry cycle.
    if state["attempts"] == 0:
        ans = "Try turning it off and on."
    else:
        ans = ask_claude("technical", state["question"],
                         "Clear your cache and reset your password via the login page.")
    return {"answer": ans, "attempts": state["attempts"] + 1,
            "log": [f"tech_agent attempt {state['attempts'] + 1}"]}


def shipping_agent(state: State) -> dict:
    ans = ask_claude("shipping", state["question"],
                     "Your package is on its way; here's your tracking link.")
    return {"answer": ans,
            "attempts": state["attempts"] + 1, "log": ["shipping_agent answered"]}


def general_agent(state: State) -> dict:
    ans = ask_claude("general", state["question"], "Thanks for reaching out! A human will follow up.")
    return {"answer": ans,
            "attempts": state["attempts"] + 1, "log": ["general_agent answered"]}


def review(state: State) -> dict:
    return {"log": [f"review saw: {state['answer']!r}"]}


# 3. ROUTING FUNCTIONS (used by conditional edges) ------------------------
def route_by_category(state: State) -> Literal["billing", "tech", "shipping", "general"]:
    return state["category"]


def good_enough(state: State) -> Literal["retry", "done"]:
    weak = "off and on" in state["answer"]
    return "retry" if weak and state["attempts"] < 3 else "done"


# 4. BUILD THE GRAPH -------------------------------------------------------
builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing", billing_agent)
builder.add_node("tech", tech_agent)
builder.add_node("shipping", shipping_agent)
builder.add_node("general", general_agent)
builder.add_node("review", review)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_category)  # return value = next node
for agent in ("billing", "tech", "shipping", "general"):
    builder.add_edge(agent, "review")
builder.add_conditional_edges("review", good_enough, {"retry": "tech", "done": END})

# 5. COMPILE WITH MEMORY ----------------------------------------------------
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    print("Graph (paste into mermaid.live to see it):\n")
    print(graph.get_graph().draw_mermaid())

    questions = [
        "I was charged twice, can I get a refund?",
        "The app shows an error on login",
        "Where is my package? The delivery is late.",
        "What are your opening hours?",
    ]
    for i, q in enumerate(questions):
        config = {"configurable": {"thread_id": f"ticket-{i}"}}
        print(f"\n=== {q}")
        # stream() yields each node's update as it runs
        for step in graph.stream({"question": q}, config, stream_mode="updates"):
            node, update = next(iter(step.items()))
            print(f"  [{node}] {update.get('log')}")
        final = graph.get_state(config).values  # read saved state from the checkpointer
        print(f"  ANSWER: {final['answer']}  (attempts={final['attempts']})")

