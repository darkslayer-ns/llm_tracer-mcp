"""
MCP Server Process Management.

This module handles the lifecycle management of MCP server processes,
including starting, stopping, monitoring, and script generation.
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("llm-mcp-process")


class MCPServerProcess:
    """
    Manages MCP server process lifecycle.
    
    This class handles server process starting/stopping, process monitoring,
    script generation, and resource cleanup.
    """
    
    def __init__(self, server_name: str = "llm-mcp-framework", semantic_config: Optional[Any] = None):
        """
        Initialize the server process manager.
        
        Args:
            server_name: Name of the server for identification
            semantic_config: Semantic search configuration for the server
        """
        self.server_name = server_name
        self.semantic_config = semantic_config
        self.process: Optional[subprocess.Popen] = None
        self.server_script_path: Optional[Path] = None
        self._setup_server_script()
    
    def _setup_server_script(self) -> None:
        """Setup the server runner script path."""
        self.server_script_path = Path(__file__).parent / "run_server.py"
    
    def create_server_runner_script(self) -> None:
        """Create the server runner script if it doesn't exist."""
        if not self.server_script_path or self.server_script_path.exists():
            return
        
        # Ensure parent directory exists
        self.server_script_path.parent.mkdir(parents=True, exist_ok=True)
        
        runner_code = '''#!/usr/bin/env python3
"""MCP Server runner script."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the server
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_mcp.mcp.server.server import FastMCPServer

def main():
    """Run the FastMCP server."""
    server = FastMCPServer()
    server.run()

if __name__ == "__main__":
    main()
'''
        
        self.server_script_path.write_text(runner_code)
        self.server_script_path.chmod(0o755)
        logger.info(f"Created server runner script: {self.server_script_path}")
    
    def start_server_process(self) -> bool:
        """
        Start the MCP server as a subprocess.
        
        Returns:
            True if server started successfully
        """
        if self.is_running():
            logger.info("MCP server process already running")
            return True
        
        if not self.server_script_path:
            logger.error("Server script path not configured")
            return False
        
        # Create server script if needed
        self.create_server_runner_script()
        
        if not self.server_script_path.exists():
            logger.error(f"Server script not found: {self.server_script_path}")
            return False
        
        try:
            # Prepare command arguments
            cmd_args = [sys.executable, str(self.server_script_path)]
            
            # Add qdrant location if semantic config is provided
            if self.semantic_config and hasattr(self.semantic_config, 'qdrant_config'):
                qdrant_location = self.semantic_config.qdrant_config.location
                if qdrant_location and qdrant_location != ":memory:":
                    # Ensure the database directory exists for validation
                    from pathlib import Path
                    db_path = Path(qdrant_location)
                    if not db_path.exists():
                        db_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created database directory: {qdrant_location}")
                    
                    # Verify the path is accessible
                    if not db_path.exists():
                        logger.error(f"Failed to create database directory: {qdrant_location}")
                    else:
                        logger.info(f"Verified database directory exists: {qdrant_location}")
                    
                    cmd_args.extend(["--qdrant-location", qdrant_location])
                    
                    # Add collection name and vector size
                    collection_name = self.semantic_config.qdrant_config.collection_name
                    vector_size = self.semantic_config.qdrant_config.vector_size
                    cmd_args.extend(["--collection-name", collection_name])
                    cmd_args.extend(["--vector-size", str(vector_size)])
                    
                    logger.info(f"Adding semantic config arguments: location={qdrant_location}, collection={collection_name}, vector_size={vector_size}")
            
            logger.info(f"Starting MCP server process: {' '.join(cmd_args)}")
            self.process = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"MCP server started with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server process: {e}")
            self.process = None
            return False
    
    def stop_server_process(self, timeout: float = 5.0) -> bool:
        """
        Stop the MCP server process.
        
        Args:
            timeout: Timeout for graceful shutdown
            
        Returns:
            True if server stopped successfully
        """
        if not self.process:
            logger.info("No MCP server process to stop")
            return True
        
        try:
            logger.info(f"Stopping MCP server process (PID: {self.process.pid})")
            
            # Try graceful termination first
            self.process.terminate()
            
            try:
                self.process.wait(timeout=timeout)
                logger.info("MCP server process terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()
                logger.info("MCP server process killed")
            
            self.process = None
            return True
            
        except Exception as e:
            logger.error(f"Error stopping MCP server process: {e}")
            return False
    
    def is_running(self) -> bool:
        """
        Check if the server process is running.
        
        Returns:
            True if process is running
        """
        if not self.process:
            return False
        
        # Check if process is still alive
        return self.process.poll() is None
    
    def get_process_id(self) -> Optional[int]:
        """
        Get the process ID of the running server.
        
        Returns:
            Process ID or None if not running
        """
        if self.is_running() and self.process:
            return self.process.pid
        return None
    
    def get_process_status(self) -> dict:
        """
        Get detailed process status information.
        
        Returns:
            Dictionary with process status details
        """
        return {
            "server_name": self.server_name,
            "is_running": self.is_running(),
            "process_id": self.get_process_id(),
            "script_path": str(self.server_script_path) if self.server_script_path else None,
            "script_exists": self.server_script_path.exists() if self.server_script_path else False
        }
    
    async def ensure_server_running(self, start_delay: float = 1.0) -> bool:
        """
        Ensure the MCP server is running, starting it if necessary.
        
        Args:
            start_delay: Delay after starting server to allow initialization
            
        Returns:
            True if server is running
        """
        if self.is_running():
            return True
        
        if self.start_server_process():
            # Give the server a moment to start
            await asyncio.sleep(start_delay)
            return self.is_running()
        
        return False
    
    def restart_server_process(self, timeout: float = 5.0, start_delay: float = 1.0) -> bool:
        """
        Restart the MCP server process.
        
        Args:
            timeout: Timeout for stopping the current process
            start_delay: Delay after starting the new process
            
        Returns:
            True if restart was successful
        """
        logger.info("Restarting MCP server process")
        
        # Stop current process
        if not self.stop_server_process(timeout):
            logger.error("Failed to stop current server process")
            return False
        
        # Start new process
        if not self.start_server_process():
            logger.error("Failed to start new server process")
            return False
        
        # Wait for startup
        import time
        time.sleep(start_delay)
        
        if self.is_running():
            logger.info("MCP server process restarted successfully")
            return True
        else:
            logger.error("MCP server process failed to start after restart")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources and stop the server process."""
        if self.is_running():
            self.stop_server_process()
        
        logger.debug("MCPServerProcess cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup