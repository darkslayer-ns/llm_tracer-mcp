"""
Colored logging utility for LLM-MCP framework.
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels and message types."""
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        
        # Standard colors
        'BLACK': '\033[30m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        
        # Bright colors
        'BRIGHT_BLACK': '\033[90m',
        'BRIGHT_RED': '\033[91m',
        'BRIGHT_GREEN': '\033[92m',
        'BRIGHT_YELLOW': '\033[93m',
        'BRIGHT_BLUE': '\033[94m',
        'BRIGHT_MAGENTA': '\033[95m',
        'BRIGHT_CYAN': '\033[96m',
        'BRIGHT_WHITE': '\033[97m',
    }
    
    # Message type colors
    MESSAGE_COLORS = {
        'TX_LLM': COLORS['BRIGHT_BLUE'],      # Transmit to LLM (blue)
        'RX_LLM': COLORS['BRIGHT_MAGENTA'],   # Receive from LLM (magenta)
        'LLM_RESPONSE': COLORS['BRIGHT_CYAN'], # Final LLM response (cyan)
        'TOOL_CALL': COLORS['BRIGHT_YELLOW'], # Tool calls (yellow)
        'TOOL_RESULT': COLORS['BRIGHT_GREEN'], # Tool results (green)
        'MCP_SERVER': COLORS['MAGENTA'],      # MCP server messages (magenta)
        'ERROR': COLORS['BRIGHT_RED'],        # Errors (red)
        'INFO': COLORS['WHITE'],              # General info (white)
        'DEBUG': COLORS['BRIGHT_BLACK'],      # Debug (gray)
    }
    
    def format(self, record):
        # Get message type from record if available
        msg_type = getattr(record, 'msg_type', 'INFO')
        color = self.MESSAGE_COLORS.get(msg_type, self.COLORS['WHITE'])
        
        # Format the message
        formatted = super().format(record)
        
        # Add color
        return f"{color}{formatted}{self.COLORS['RESET']}"


class LLMMCPLogger:
    """Centralized logger for LLM-MCP framework with colored output."""
    
    def __init__(self, name: str = "llm-mcp", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler with colored formatter
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _log_with_type(self, level: int, msg_type: str, message: str):
        """Log a message with a specific type for coloring."""
        extra = {'msg_type': msg_type}
        self.logger.log(level, message, extra=extra)
    
    def tx_llm(self, message: str):
        """Log message being sent to LLM (blue)."""
        self._log_with_type(logging.INFO, 'TX_LLM', f"📤 TX → LLM: {message}")
    
    def rx_llm(self, message: str):
        """Log message received from LLM (magenta)."""
        self._log_with_type(logging.INFO, 'RX_LLM', f"📥 RX ← LLM: {message}")
    
    def llm_response(self, message: str):
        """Log final LLM response (cyan)."""
        self._log_with_type(logging.INFO, 'LLM_RESPONSE', f"🤖 LLM: {message}")
    
    def tool_call(self, tool_name: str, arguments: dict):
        """Log tool call (yellow)."""
        args_str = str(arguments) if len(str(arguments)) < 100 else f"{str(arguments)[:97]}..."
        self._log_with_type(logging.INFO, 'TOOL_CALL', f"🔧 TOOL CALL: {tool_name}({args_str})")
    
    def tool_result(self, tool_name: str, success: bool, result_preview: Optional[str] = None):
        """Log tool result (cyan)."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        msg = f"🔧 TOOL RESULT: {tool_name} → {status}"
        if result_preview:
            preview = result_preview if len(result_preview) < 100 else f"{result_preview[:97]}..."
            msg += f" | {preview}"
        self._log_with_type(logging.INFO, 'TOOL_RESULT', msg)
    
    def mcp_server(self, message: str):
        """Log MCP server activity (magenta)."""
        self._log_with_type(logging.INFO, 'MCP_SERVER', f"🔌 MCP: {message}")
    
    def error(self, message: str):
        """Log error (red)."""
        self._log_with_type(logging.ERROR, 'ERROR', f"❌ ERROR: {message}")
    
    def info(self, message: str):
        """Log general info (white)."""
        self._log_with_type(logging.INFO, 'INFO', f"ℹ️  INFO: {message}")
    
    def debug(self, message: str):
        """Log debug info (gray)."""
        self._log_with_type(logging.DEBUG, 'DEBUG', f"🐛 DEBUG: {message}")
    
    def session_start(self, provider: str, model: str):
        """Log session start."""
        self.info(f"Session started with {provider}/{model}")
    
    def session_end(self):
        """Log session end."""
        self.info("Session ended")
    
    def available_tools(self, tools: list):
        """Log available tools."""
        tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
        self.info(f"Available tools: {', '.join(tool_names)}")


# Global logger instance
llm_logger = LLMMCPLogger()


def get_logger() -> LLMMCPLogger:
    """Get the global LLM-MCP logger instance."""
    return llm_logger


def set_log_level(level: int):
    """Set the logging level."""
    llm_logger.logger.setLevel(level)