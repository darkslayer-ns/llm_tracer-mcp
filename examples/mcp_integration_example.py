"""
Example demonstrating MCP server and client integration.

This example shows how the framework automatically spawns an MCP server
when imported and connects to it via the MCP client for tool execution.
"""

import asyncio
import os
from llm_mcp import LLMSession, GenericRequest, FastMCPServer, MCPManager


async def mcp_integration_example():
    """Example showing MCP server and client integration."""
    print("🔧 MCP Integration Example")
    print("=" * 50)
    
    # The MCP server will be automatically started when we create the session
    session = LLMSession(
        provider="openai",
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        # No need to specify tools - they come from the MCP server
    )
    
    # Create a request that will use MCP tools
    request = GenericRequest(
        system_prompt="""You are a helpful code analysis assistant. You have access to file reading and search tools via MCP.
        
        Use the available tools to:
        1. Read and analyze Python files
        2. Search for specific patterns
        3. Provide insights about code structure
        
        Always use tools to gather information before providing analysis.""",
        
        user_message="Please analyze the Python files in the current directory. Look for any interesting patterns, imports, or code structure. Use the available tools to read files and search for patterns.",
        
        context={
            "task": "code_analysis",
            "target_directory": ".",
            "focus": "Python files"
        }
    )
    
    try:
        async with session:
            print("🔄 Starting analysis with MCP tools...")
            print("📡 MCP server should be automatically started")
            print("🔗 MCP client will connect to the server")
            print()
            
            response = await session.run_analysis(request)
            
            if response.success:
                print("✅ Analysis completed!")
                print(f"\n📊 Model used: {response.metadata.model_used}")
                print(f"🔧 Tools used: {', '.join(response.metadata.tools_used) if response.metadata.tools_used else 'None'}")
                print(f"⏱️  Duration: {response.metadata.total_duration:.2f}s")
                print(f"\n📝 Analysis Result:")
                print("-" * 50)
                print(response.raw_content)
            else:
                print(f"❌ Error: {response.error.message}")
                if response.error.suggestion:
                    print(f"💡 Suggestion: {response.error.suggestion}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")


async def direct_mcp_client_example():
    """Example using MCP client directly."""
    print("\n🔧 Direct MCP Client Example")
    print("=" * 50)
    
    # Use MCP manager directly
    async with MCPManager() as mcp_manager:
        print("✅ Connected to MCP server")
        
        # Get available tools
        tools = mcp_manager.get_available_tools()
        print(f"📋 Available tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool['function']['name']}: {tool['function']['description']}")
        
        # Execute a tool directly
        from llm_mcp.models.tools import ToolCall
        
        tool_call = ToolCall(
            tool_name="read_file",
            arguments={"file_path": "llm_mcp/__init__.py"}
        )
        
        print(f"\n🔧 Executing tool: {tool_call.tool_name}")
        result = await mcp_manager.execute_tool_call(tool_call)
        
        if result.success:
            print("✅ Tool execution successful!")
            print(f"📄 Result preview: {str(result.data)[:200]}...")
        else:
            print(f"❌ Tool execution failed: {result.error.message}")


async def mcp_server_standalone_example():
    """Example running MCP server standalone."""
    print("\n🖥️  MCP Server Standalone Example")
    print("=" * 50)
    print("This would run the MCP server directly (commented out to avoid blocking)")
    print("To run standalone server:")
    print("  python -m llm_mcp.server.run_server")
    print()
    
    # Uncomment to run server directly (will block)
    # server = FastMCPServer()
    # await server.run()


async def main():
    """Run MCP integration examples."""
    print("🚀 MCP Integration Examples")
    print("=" * 60)
    print("Demonstrating MCP server and client integration")
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable to run LLM examples")
        print("   export OPENAI_API_KEY='your-key-here'")
        print()
        print("🔧 Running MCP-only examples...")
        await direct_mcp_client_example()
        await mcp_server_standalone_example()
    else:
        # Run all examples
        await mcp_integration_example()
        await direct_mcp_client_example()
        await mcp_server_standalone_example()
    
    print("\n🎉 MCP integration examples completed!")
    print("The framework now has full MCP server and client support!")


if __name__ == "__main__":
    asyncio.run(main())