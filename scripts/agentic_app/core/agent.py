from typing import List, Dict, Any, Union, TypedDict
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.llms.base import LLM
from ..utils.helpers import logger
from ..tools.graph_query_tools import GraphQueryTools
import re
import difflib

class GraphState(TypedDict):
    messages: List[dict]
    tools: Dict[str, Any]
    tool_results: Dict[str, Any]
    current_tool: str

def agent_node_fn(state: GraphState, llm: LLM) -> GraphState:
    """
    Combine the list of messages into a single string, then call llm.invoke(...)
    and append the result as an AIMessage.
    """
    conversation = state["messages"]
    if not conversation:
        return state

    # Build one big prompt from the messages
    prompt_parts = []
    for msg in conversation:
        if isinstance(msg, SystemMessage):
            prompt_parts.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            prompt_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_parts.append(f"Assistant: {msg.content}")
            # If there's a tool result after an AI message with a tool call,
            # include it in the conversation
            if "TOOL:" in msg.content and state.get("tool_results"):
                tool_name = msg.content.split("TOOL:")[1].strip().split()[0]
                if tool_name in state["tool_results"]:
                    prompt_parts.append(f"Tool Result: {state['tool_results'][tool_name]}")
        else:
            # In case any other message
            role = msg.get("role", "Unknown")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
    # Join them with newlines
    prompt_text = "\n".join(prompt_parts)

    # LOG the prompt before calling the LLM
    logger.debug("Agent node: sending this prompt to the LLM:\n%s", prompt_text)

    try:
        # Call LLM
        llm_output = llm.invoke(prompt_text)
        
        if not llm_output or not isinstance(llm_output, str):
            logger.error(f"Invalid LLM output type: {type(llm_output)}")
            raise ValueError("LLM returned invalid output")
            
        if len(llm_output.strip()) == 0:
            logger.error("LLM returned empty response")
            raise ValueError("LLM returned empty response")

        # LOG the response from the LLM
        logger.debug("Agent node: received this response from the LLM:\n%s", llm_output)
        
        # Check if this is a response to a human query and should contain a tool call
        is_tool_call_expected = False
        for i, msg in enumerate(conversation):
            if isinstance(msg, HumanMessage) and i == len(conversation) - 1:
                # This is the last message and it's from a human, so we expect a tool call
                is_tool_call_expected = True
                break
        
        # If we expect a tool call but don't see one, try to extract it or force one
        if is_tool_call_expected and "TOOL:" not in llm_output:
            logger.warning("LLM did not return a tool call when one was expected")
            
            # Get the human query
            human_query = conversation[-1].content.lower()
            
            # Try to determine the appropriate tool based on the query
            if "shortest path" in human_query or "best route" in human_query:
                # Extract airport codes
                airports = re.findall(r'\b([A-Z]{3})\b', human_query.upper())
                if len(airports) >= 2:
                    llm_output = f"TOOL: FindShortestPath {airports[0]},{airports[1]}"
                    logger.info(f"Forced tool call: {llm_output}")
            elif "direct flight" in human_query or "direct flights" in human_query:
                # Extract airport code
                airports = re.findall(r'\b([A-Z]{3})\b', human_query.upper())
                if airports:
                    llm_output = f"TOOL: FindDirectFlights {airports[0]}"
                    logger.info(f"Forced tool call: {llm_output}")
            elif "importance" in human_query or "analyze" in human_query:
                # Extract airport code
                airports = re.findall(r'\b([A-Z]{3})\b', human_query.upper())
                if airports:
                    llm_output = f"TOOL: AnalyzeAirportImportance {airports[0]}"
                    logger.info(f"Forced tool call: {llm_output}")
        
        # Append as AIMessage
        state["messages"].append(AIMessage(content=llm_output))
        return state
        
    except Exception as e:
        logger.error(f"Error in agent_node_fn: {str(e)}")
        raise

def create_agent_graph(llm: LLM, tools: List[BaseTool]) -> StateGraph:
    """Create the agent graph."""
    # Define the state type
    workflow = StateGraph(GraphState)
    
    # Create a mapping of tool name to tool
    tool_map = {tool.name: tool for tool in tools}
    
    # Define the agent node
    def agent_node(state: GraphState) -> GraphState:
        """Execute the agent node."""
        # Initialize state if needed
        if "tools" not in state:
            state["tools"] = tool_map
        if "tool_results" not in state:
            state["tool_results"] = {}
        if "current_tool" not in state:
            state["current_tool"] = ""
        if "messages" not in state:
            state["messages"] = []
            
        # Execute the agent
        return agent_node_fn(state, llm)
    
    # Define the tool executor node
    def tool_executor_node(state: GraphState) -> GraphState:
        """Execute the tool node."""
        try:
            # Get the last message
            last_msg = state["messages"][-1]
            if not isinstance(last_msg, AIMessage) or "TOOL:" not in last_msg.content:
                logger.error("No tool call found in the last message")
                state["messages"].append(SystemMessage(content="Error: No tool call found."))
                return state
            
            # Extract the tool call
            content = last_msg.content
            tool_call_match = re.search(r"TOOL:\s*(\w+)(?:\s+(.+))?", content)
            if not tool_call_match:
                logger.error("Failed to parse tool call")
                state["messages"].append(SystemMessage(content="Error: Failed to parse tool call."))
                return state
            
            tool_name = tool_call_match.group(1)
            tool_input = tool_call_match.group(2) if tool_call_match.group(2) else ""
            
            # Execute the tool
            if tool_name in state["tools"]:
                logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                tool_result = state["tools"][tool_name].func(tool_input)
                
                # Store the result
                state["tool_results"][tool_name] = tool_result
                
                # Add the result to the messages
                state["messages"].append(SystemMessage(content=f"Tool Result: {tool_result}"))
            else:
                logger.error(f"Unknown tool: {tool_name}")
                state["messages"].append(SystemMessage(content=f"Error: Unknown tool '{tool_name}'."))
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            state["messages"].append(SystemMessage(content=f"Error executing tool: {str(e)}"))
        
        return state
    
    # Add nodes to the graph
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_executor", tool_executor_node)
    
    # Define the conditional edges
    def conditional_route(state: GraphState) -> Dict[str, float]:
        """Route to the next node based on the last message."""
        if not state.get("messages"):
            return {"agent": 1.0}
        
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            content = last_msg.content
            
            # Check for tool calls
            if "TOOL:" in content:
                try:
                    # Extract the tool call
                    tool_call_match = re.search(r"TOOL:\s*(\w+)(?:\s+(.+))?", content)
                    if tool_call_match:
                        tool_name = tool_call_match.group(1)
                        tool_input = tool_call_match.group(2) if tool_call_match.group(2) else ""
                        
                        # Validate the tool name
                        if tool_name in state["tools"]:
                            logger.debug(f"Routing - Valid tool call found: TOOL: {tool_name} {tool_input}")
                            
                            # Update the message to contain only the tool call
                            state["messages"][-1] = AIMessage(content=f"TOOL: {tool_name} {tool_input}")
                            
                            # Set the current tool
                            state["current_tool"] = tool_name
                            return {"tool_executor": 1.0}
                except Exception as e:
                    logger.error(f"Error in routing: {str(e)}")
                    state["messages"].append(SystemMessage(content=f"Error: {str(e)}"))
        
        return {"agent": 1.0}
    
    # Add edges
    workflow.add_edge("agent", conditional_route)
    workflow.add_edge("tool_executor", lambda x: {"agent": 1.0})
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app 