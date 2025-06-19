#!/usr/bin/env python3
"""
Example demonstrating the multi-file reading capability of the FileReadTool.

This example shows how to:
1. Read multiple files at once
2. Use focus lines to highlight specific lines
3. Use line ranges to read only portions of files
4. Handle mixed success/failure scenarios
"""

import asyncio
from llm_mcp.tools.filesystem.file_ops import FileReadTool, FileReadRequest, FileSpec
from llm_mcp.models.tools import ToolExecutionContext


async def demonstrate_multi_file_reading():
    """Demonstrate various multi-file reading scenarios."""
    
    # Initialize the tool and context
    tool = FileReadTool()
    context = ToolExecutionContext(
        session_id="demo_session",
        request_id="demo_request"
    )
    
    print("🔍 Multi-File Reading Demonstration")
    print("=" * 50)
    
    # Example 1: Read multiple Python files from the project
    print("\n📁 Example 1: Reading multiple project files")
    request = FileReadRequest(
        files=[
            FileSpec(file_path="llm_mcp/__init__.py"),
            FileSpec(file_path="llm_mcp/models/base.py", line_range=(1, 20)),
            FileSpec(file_path="llm_mcp/utils/constants.py", focus_lines=[1, 5, 10])
        ]
    )
    
    response = await tool.execute(request, context)
    print(f"✅ Success: {response.success}")
    print(f"📊 Files processed: {response.total_files}")
    print(f"✅ Successful: {response.successful_files}")
    print(f"❌ Failed: {response.failed_files}")
    
    for i, file_result in enumerate(response.files, 1):
        print(f"\n📄 File {i}: {file_result.file_path}")
        print(f"   Status: {'✅ Success' if file_result.success else '❌ Failed'}")
        if file_result.success:
            print(f"   Lines: {file_result.line_count}")
            print(f"   Size: {file_result.size_bytes} bytes")
            if file_result.highlighted_lines:
                print(f"   Highlighted: {file_result.highlighted_lines}")
            # Show first few lines of content
            if file_result.content:
                lines = file_result.content.split('\n')[:3]
                print(f"   Preview: {lines[0][:60]}...")
        else:
            print(f"   Error: {file_result.error_message}")
    
    # Example 2: Reading configuration and documentation files
    print("\n\n📋 Example 2: Reading config and docs")
    request = FileReadRequest(
        files=[
            FileSpec(file_path="pyproject.toml", line_range=(1, 15)),
            FileSpec(file_path="README.md", line_range=(1, 10)),
            FileSpec(file_path=".gitignore", focus_lines=[1, 10, 20])
        ]
    )
    
    response = await tool.execute(request, context)
    print(f"✅ Success: {response.success}")
    
    for file_result in response.files:
        if file_result.success:
            print(f"\n📄 {file_result.file_path} ({file_result.line_count} lines)")
            # Show content with line numbers if it's short
            if file_result.line_count <= 15:
                print("Content:")
                for i, line in enumerate(file_result.content.split('\n'), 1):
                    print(f"  {i:2d}: {line}")
        else:
            print(f"\n❌ Failed to read {file_result.file_path}: {file_result.error_message}")
    
    # Example 3: Efficient code review - read related files
    print("\n\n🔍 Example 3: Code review - reading related files")
    request = FileReadRequest(
        files=[
            FileSpec(
                file_path="llm_mcp/tools/filesystem/file_ops.py",
                focus_lines=[16, 17, 18, 19, 20],  # FileSpec class definition
                line_range=(15, 25)
            ),
            FileSpec(
                file_path="llm_mcp/tools/filesystem/__init__.py",
                line_range=(1, 10)
            )
        ]
    )
    
    response = await tool.execute(request, context)
    
    if response.success:
        print("✅ Successfully read related files for code review")
        for file_result in response.files:
            print(f"\n📄 {file_result.file_path}")
            if file_result.highlighted_lines:
                print(f"🎯 Focus lines: {file_result.highlighted_lines}")
            print("Content:")
            print(file_result.content)
    
    print("\n" + "=" * 50)
    print("🎉 Multi-file reading demonstration complete!")
    print("\n💡 Key benefits:")
    print("   • Read multiple files in a single request")
    print("   • Focus on specific lines across files")
    print("   • Read only relevant portions with line ranges")
    print("   • Efficient for code analysis and review")
    print("   • Graceful handling of missing files")


if __name__ == "__main__":
    asyncio.run(demonstrate_multi_file_reading())