"""
MCP (Model Context Protocol) functionality for the LLM-MCP framework.

This module contains all MCP-related components organized with clear separation of concerns:
- client: Pure FastMCP client for server connections
- manager: High-level MCP management and orchestration
- server: MCP server implementation components
- tools: Tool adaptation for MCP usage
"""

from .client import FastMCPClient
from .manager import MCPManager

__all__ = [
    "FastMCPClient",
    "MCPManager",
]