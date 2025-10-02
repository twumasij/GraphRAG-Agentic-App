Below is one way to fix the “Invalid input type <class 'dict'>” error so your code can run. The core issue is that your LLM node is being passed the entire state dictionary rather than a valid prompt (which must be a string or a list of LangChain BaseMessages).

In LangGraph, simply adding the raw LLM object (workflow.add_node("agent", llm)) causes the state (which is a dict) to be forwarded directly to llm(). However, LangChain’s LLM.__call__ expects either:

A single string prompt, or
A list of LangChain BaseMessage objects (e.g. [SystemMessage(...), HumanMessage(...)]), or
A special LangChain PromptValue object.
Below is an example of how to fix this by creating a small custom node (or “callable”) that:

Extracts the conversation from state["messages"] (which is already a list of SystemMessage/HumanMessage/AIMessage),
Passes that list to your llm,
Appends the LLM’s result as an AIMessage,
Continues to the next step (possibly tool usage).
You can adapt this pattern to your own logic as needed.

1) Create a custom agent node
python
Copy
Edit
from langchain.schema import AIMessage, BaseMessage  # for clarity
from typing import Union, List

def agent_node_fn(state: GraphState, llm: LLM) -> GraphState:
    """
    A custom 'agent' node for LangGraph that calls the LLM with the user messages.
    """
    # 1) Extract all messages from the state
    conversation: List[BaseMessage] = state["messages"]

    # 2) Call the LLM using the entire conversation as input
    #    The LLM returns a string
    llm_output: str = llm(conversation)

    # 3) Append an AIMessage with the LLM output
    state["messages"].append(AIMessage(content=llm_output))

    # 4) (Optional) parse out any "tool calls" from llm_output if using ReAct-like approaches
    # For example, you might set something like:
    # state["current_tool"] = parse_tool_name_from(llm_output)
    # or state["tool_results"]["some_tool"] = ...
    #
    # For now, let's keep it simple: we won't parse any tool calls.

    return state
This function does all of the following:

Looks inside your state["messages"] (the conversation).
Calls the LLM with those messages (a valid input type).
Stores the LLM output in AIMessage(content=...).
Returns the updated state so LangGraph can continue.
2) Update your graph creation
In your create_agent_graph function, do not directly add the LLM as a node. Instead, wrap the LLM call inside the custom function you just wrote:

python
Copy
Edit
def create_agent_graph(llm: LLM, tools: List[BaseTool]) -> StateGraph:
    tool_node = ToolNode(tools)
    workflow = StateGraph(GraphState)

    # Create the 'agent' node by wrapping the LLM in our custom function
    def agent_node(state: GraphState) -> GraphState:
        return agent_node_fn(state, llm)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_executor", tool_node)

    # Routing function to decide where to go next
    def route(state: GraphState) -> Union[str, List[str]]:
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last AIMessage doesn't contain a "tool request," we can stop.
        # For now, let's just end if there's no more user question.
        if isinstance(last_message, AIMessage):
            # Example placeholder: if "TOOL:" is in the content, go to tool_executor
            if "TOOL:" in last_message.content:
                return "tool_executor"
            else:
                return END
        
        # If there's no AIMessage or no tool call, end by default.
        return END

    # Add edges
    workflow.add_conditional_edges(
        "agent",
        route,
        {
            "tool_executor": "tool_executor",
            END: END
        }
    )

    # Add edge from tools back to agent
    workflow.add_edge("tool_executor", "agent")

    # Set entry point
    workflow.set_entry_point("agent")

    return workflow.compile()
What changed?

workflow.add_node("agent", agent_node) references your custom function that calls the LLM correctly.
The agent_node_fn is invoked, which does llm(conversation) instead of llm(state).
3) Invoke the graph with the correct initial state
Your main() can stay roughly the same, but make sure you pass valid BaseMessages in state["messages"]. For example:

python
Copy
Edit
def main():
    # (health checks + flight tools omitted for brevity)

    # Create tool list for agent with refined descriptions
    tool_list = [
        Tool(
            name="FindDirectFlights",
            func=tools.find_direct_flights,
            description="Returns all direct flights from a given airport. Input: IATA code (e.g., 'LAX')."
        ),
        Tool(
            name="FindShortestPath",
            func=tools.find_shortest_path,
            description="Finds the shortest path between two airports. Input: 'LAX,JFK'."
        ),
        Tool(
            name="AnalyzeAirport",
            func=tools.analyze_airport_importance,
            description="Analyzes the importance of an airport using network metrics. Input: IATA code (e.g., 'LAX')."
        ),
    ]

    # Initialize Ollama-based LLM
    llm = initialize_llm()

    # Create the agent graph
    agent_executor = create_agent_graph(llm, tool_list)

    print("\nFlight Network Analysis System Ready!")
    print("Type 'quit' to exit\n")

    while True:
        query = input("What would you like to know? ").strip()
        
        if query.lower() == 'quit':
            break
        
        # Create a new state with messages for each query
        state = {
            "messages": [
                SystemMessage(
                    content="You are a helpful assistant that analyzes flight networks. "
                            "Use the provided tools to answer questions about flights, routes, "
                            "and airport importance. Use IATA codes in your answers."
                ),
                HumanMessage(content=query)
            ],
            "tools": {tool.name: tool for tool in tool_list},
            "tool_results": {},
            "current_tool": None
        }
        
        # Run the graph
        try:
            result = agent_executor.invoke(state)
            # 'result' is the final updated state
            # The last message in result["messages"] should be the AI response
            messages = result["messages"]
            ai_responses = [m for m in messages if isinstance(m, AIMessage)]
            if ai_responses:
                print("\nAssistant:", ai_responses[-1].content)
            else:
                print("\n(No AI response generated.)")

        except Exception as e:
            print(f"Error: {str(e)}")
Here, each iteration:

We build state with a fresh conversation [SystemMessage(...), HumanMessage(...)].
We call agent_executor.invoke(state).
That calls your custom agent_node_fn, which calls the LLM with the conversation.
The LLM result is appended as an AIMessage to state["messages"].
The final state is returned, from which we can print the last AI response.
4) Confirm or extend for tool usage
If you also want the LLM to call the flight tools automatically (ReAct style), you would need to:

Teach the LLM a format for calling tools (e.g., “Use the following format for a tool call: Tool: <ToolName>\nTool Input: ... … etc.”).
Parse any actual tool calls from the LLM’s output (like using a regex).
Invoke the relevant tool within your ToolNode or the routing function.
The snippet above sets up a minimal flow:

The LLM runs,
We check if the AI message indicates a tool call (e.g., if it contains TOOL:),
If so, we route to "tool_executor", which executes the tool and returns the tool’s result to the conversation,
Then we go back to "agent" to let the LLM incorporate the tool result.
The key fix, however, is ensuring that when the “agent” node calls the LLM, it is passing a valid message list instead of the entire state dictionary. That fixes the Invalid input type <class 'dict'> error.

