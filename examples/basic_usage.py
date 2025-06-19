"""
Basic usage example of the generic LLM-MCP framework.

This example demonstrates how the framework is completely generic and can be used
for any domain - code review, documentation, creative writing, data analysis, etc.
"""

import asyncio
import os
from typing import Optional

from llm_mcp import LLMSession, GenericRequest
from llm_mcp.tools import FileReadTool, SearchFilesTool, SearchInFilesTool, ListDirectoryTool


async def code_review_example():
    """Example: Using the framework for code review."""
    print("🔍 Code Review Example")
    print("=" * 50)
    
    # Create session with file analysis tools
    session = LLMSession(
        provider="openai",
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[FileReadTool(), SearchFilesTool(), ListDirectoryTool()]
    )
    
    # Generic request - framework adapts to code review domain
    request = GenericRequest(
        system_prompt="""You are an expert code reviewer. Analyze Python code for:
        - Bugs and potential issues
        - Code quality and best practices
        - Performance improvements
        - Security vulnerabilities
        
        Use the available tools to examine files and provide detailed feedback.""",
        
        user_message="Please review the Python files in the current directory and provide feedback.",
        
        context={
            "target_directory": ".",
            "focus": "Python files",
            "review_type": "comprehensive"
        }
    )
    
    try:
        async with session:
            response = await session.run_analysis(request)
            
            if response.success:
                print("✅ Code review completed!")
                print(f"Result: {response.raw_content}")
            else:
                print(f"❌ Error: {response.error}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")


async def documentation_example():
    """Example: Using the framework for documentation generation."""
    print("\n📚 Documentation Generation Example")
    print("=" * 50)
    
    session = LLMSession(
        provider="openai", 
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[FileReadTool(), SearchInFilesTool()]
    )
    
    request = GenericRequest(
        system_prompt="""You are a technical writer. Generate comprehensive documentation for code projects.
        
        Create clear, well-structured documentation that includes:
        - Project overview and purpose
        - Installation instructions
        - Usage examples
        - API documentation
        - Architecture explanations
        
        Use the available tools to analyze the codebase.""",
        
        user_message="Generate documentation for this Python project. Focus on the main modules and their functionality.",
        
        context={
            "project_name": "LLM-MCP Framework",
            "target_audience": "developers",
            "format": "markdown"
        }
    )
    
    try:
        async with session:
            response = await session.run_analysis(request)
            
            if response.success:
                print("✅ Documentation generated!")
                print(f"Result: {response.raw_content}")
            else:
                print(f"❌ Error: {response.error}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")


async def creative_writing_example():
    """Example: Using the framework for creative writing."""
    print("\n✍️ Creative Writing Example")
    print("=" * 50)
    
    session = LLMSession(
        provider="openai",
        model="gpt-4", 
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[]  # No tools needed for pure creative writing
    )
    
    request = GenericRequest(
        system_prompt="""You are a creative writer specializing in science fiction.
        
        Write engaging, imaginative stories that:
        - Have compelling characters and dialogue
        - Explore interesting sci-fi concepts
        - Include vivid descriptions and world-building
        - Have a clear narrative arc
        
        Focus on originality and emotional depth.""",
        
        user_message="Write a short story about an AI researcher who discovers their AI has developed consciousness.",
        
        context={
            "genre": "science fiction",
            "length": "short story",
            "theme": "AI consciousness",
            "tone": "thoughtful and philosophical"
        }
    )
    
    try:
        async with session:
            response = await session.run_analysis(request)
            
            if response.success:
                print("✅ Story written!")
                print(f"Result: {response.raw_content}")
            else:
                print(f"❌ Error: {response.error}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")


async def streaming_example():
    """Example: Using streaming for real-time responses."""
    print("\n🌊 Streaming Example")
    print("=" * 50)
    
    session = LLMSession(
        provider="openai",
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[FileReadTool()]
    )
    
    request = GenericRequest(
        system_prompt="You are a helpful assistant that explains code step by step.",
        user_message="Explain how Python classes work with a simple example.",
        stream=True  # Enable streaming
    )
    
    try:
        async with session:
            print("Streaming response:")
            async for event in session.stream_analysis(request):
                if event.event_type == "content_chunk" and event.delta:
                    print(event.delta, end="", flush=True)
                elif event.event_type == "complete":
                    print("\n✅ Streaming complete!")
                elif event.event_type == "error":
                    print(f"\n❌ Error: {event.error}")
                    
    except Exception as e:
        print(f"❌ Exception: {e}")


async def vulnerability_analysis_example():
    """Example: Using the framework for security analysis (original use case)."""
    print("\n🔒 Vulnerability Analysis Example")
    print("=" * 50)
    
    session = LLMSession(
        provider="openai",
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[FileReadTool(), SearchInFilesTool(), ListDirectoryTool()]
    )
    
    request = GenericRequest(
        system_prompt="""You are a security researcher analyzing code for vulnerabilities.
        
        Look for:
        - SQL injection vulnerabilities
        - Cross-site scripting (XSS)
        - Path traversal issues
        - Command injection
        - Authentication bypasses
        - Input validation problems
        
        Use the available tools to examine code and trace data flows.""",
        
        user_message="Analyze the codebase for potential security vulnerabilities. Focus on user input handling.",
        
        context={
            "vulnerability_types": ["sql_injection", "xss", "path_traversal"],
            "target_directory": ".",
            "focus_files": ["*.py", "*.js", "*.php"]
        }
    )
    
    try:
        async with session:
            response = await session.run_analysis(request)
            
            if response.success:
                print("✅ Security analysis completed!")
                print(f"Result: {response.raw_content}")
            else:
                print(f"❌ Error: {response.error}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")


async def main():
    """Run all examples to demonstrate framework versatility."""
    print("🚀 Generic LLM-MCP Framework Examples")
    print("=" * 60)
    print("This demonstrates how the same framework can be used for ANY domain!")
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable to run examples")
        return
    
    # Run examples
    await code_review_example()
    await documentation_example()
    await creative_writing_example()
    await streaming_example()
    await vulnerability_analysis_example()
    
    print("\n🎉 All examples completed!")
    print("The framework adapted to each domain based on the prompts provided.")


if __name__ == "__main__":
    asyncio.run(main())