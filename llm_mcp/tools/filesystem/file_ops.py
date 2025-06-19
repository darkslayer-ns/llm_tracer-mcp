"""
File operation tools with structured Pydantic responses.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, validator

from ...models.base import BaseFrameworkModel
from ...models.tools import ToolExecutionContext
from ...tool_registry import BaseTool
from ...utils.constants import should_exclude_path


def _validate_safe_path(path: str) -> str:
    """
    Validate that a path is safe and doesn't contain path traversal attempts.
    
    Args:
        path: The path to validate
        
    Returns:
        The validated path
        
    Raises:
        ValueError: If the path contains path traversal attempts or is unsafe
    """
    if not path:
        raise ValueError("Path cannot be empty")
    
    # Convert to Path object for normalization
    try:
        path_obj = Path(path)
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid path format: {e}")
    
    # Check for path traversal attempts
    path_str = str(path_obj)
    if '..' in path_obj.parts:
        raise ValueError(f"Path traversal detected in path: {path}")
    
    # Check for absolute paths that could escape sandbox
    if path_obj.is_absolute():
        # Allow absolute paths only within current working directory tree
        try:
            cwd = Path.cwd()
            path_obj.resolve().relative_to(cwd.resolve())
        except ValueError:
            raise ValueError(f"Absolute path outside working directory not allowed: {path}")
    
    # Check for dangerous path components
    dangerous_parts = {'.', '..', '~'}
    if any(part in dangerous_parts for part in path_obj.parts):
        if path_obj.parts != ('.', ):  # Allow current directory
            raise ValueError(f"Dangerous path component detected: {path}")
    
    # Resolve the path and check it's within current directory
    try:
        resolved = path_obj.resolve()
        cwd = Path.cwd().resolve()
        resolved.relative_to(cwd)
    except (ValueError, OSError):
        raise ValueError(f"Path resolves outside working directory: {path}")
    
    return path


# Input Models
class FileSpec(BaseFrameworkModel):
    """Specification for a single file to read."""
    file_path: str = Field(..., description="Path to the file to read")
    focus_lines: Optional[List[int]] = Field(None, description="Optional specific lines to highlight")
    line_range: Optional[tuple[int, int]] = Field(None, description="Optional line range (start, end) to read")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path for security."""
        return _validate_safe_path(v)


class FileReadRequest(BaseFrameworkModel):
    """Request model for reading one or multiple files."""
    files: List[FileSpec] = Field(..., description="List of files to read (can be single file)")
    encoding: str = Field("utf-8", description="File encoding")


class FileWriteRequest(BaseFrameworkModel):
    """Request model for writing files."""
    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")
    encoding: str = Field("utf-8", description="File encoding")
    create_dirs: bool = Field(True, description="Create parent directories if they don't exist")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path for security."""
        return _validate_safe_path(v)


class ListDirectoryRequest(BaseFrameworkModel):
    """Request model for listing directory contents."""
    directory_path: str = Field(".", description="Path to the directory to list")
    show_hidden: bool = Field(False, description="Whether to show hidden files")
    recursive: bool = Field(False, description="Whether to list recursively")
    pattern: Optional[str] = Field(None, description="Optional glob pattern to filter files")
    
    @validator('directory_path')
    def validate_directory_path(cls, v):
        """Validate directory path for security."""
        return _validate_safe_path(v)


# Output Models
class SingleFileResult(BaseFrameworkModel):
    """Result for reading a single file."""
    success: bool = Field(..., description="Whether reading this file was successful")
    file_path: str = Field(..., description="Path of the file")
    content: Optional[str] = Field(None, description="File content")
    line_count: Optional[int] = Field(None, description="Number of lines in the file")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    highlighted_lines: List[int] = Field(default_factory=list, description="Lines that were highlighted")
    error_message: Optional[str] = Field(None, description="Error message if reading failed")


class FileReadResponse(BaseFrameworkModel):
    """Structured response for file reading (one or multiple files)."""
    success: bool = Field(..., description="Whether the overall operation was successful")
    encoding: str = Field(..., description="File encoding used")
    files: List[SingleFileResult] = Field(..., description="Results for each file")
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: int = Field(..., description="Number of files read successfully")
    failed_files: int = Field(..., description="Number of files that failed to read")
    error_message: Optional[str] = Field(None, description="Overall error message if operation failed")


class FileWriteResponse(BaseFrameworkModel):
    """Structured response for file writing."""
    success: bool = Field(..., description="Whether the operation was successful")
    file_path: str = Field(..., description="Path of the file that was written")
    bytes_written: Optional[int] = Field(None, description="Number of bytes written")
    created_dirs: List[str] = Field(default_factory=list, description="Directories that were created")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")


class FileInfo(BaseFrameworkModel):
    """Information about a file or directory."""
    name: str = Field(..., description="File or directory name")
    path: str = Field(..., description="Full path")
    is_directory: bool = Field(..., description="Whether this is a directory")
    size_bytes: Optional[int] = Field(None, description="Size in bytes (files only)")
    modified_time: Optional[str] = Field(None, description="Last modified time (ISO format)")
    permissions: Optional[str] = Field(None, description="File permissions")


class ListDirectoryResponse(BaseFrameworkModel):
    """Structured response for directory listing."""
    success: bool = Field(..., description="Whether the operation was successful")
    directory_path: str = Field(..., description="Path of the directory that was listed")
    items: List[FileInfo] = Field(default_factory=list, description="List of files and directories")
    total_items: int = Field(0, description="Total number of items found")
    total_files: int = Field(0, description="Number of files")
    total_directories: int = Field(0, description="Number of directories")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")


# Tool Implementations
class FileReadTool(BaseTool):
    """Tool for reading file contents with structured output."""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of one or multiple files with optional line highlighting and range selection"
    
    @property
    def input_schema(self):
        return FileReadRequest
    
    @property
    def output_schema(self):
        return FileReadResponse
    
    async def execute(
        self,
        input_data: FileReadRequest,
        context: ToolExecutionContext
    ) -> FileReadResponse:
        """Execute file reading with structured response (one or multiple files)."""
        try:
            results = []
            successful_count = 0
            failed_count = 0
            
            for file_spec in input_data.files:
                result = await self._read_single_file(file_spec, input_data.encoding)
                results.append(result)
                
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1
            
            overall_success = failed_count == 0
            
            return FileReadResponse(
                success=overall_success,
                encoding=input_data.encoding,
                files=results,
                total_files=len(results),
                successful_files=successful_count,
                failed_files=failed_count,
                error_message=f"Failed to read {failed_count} files" if failed_count > 0 else None
            )
                
        except Exception as e:
            return FileReadResponse(
                success=False,
                encoding=input_data.encoding,
                files=[],
                total_files=0,
                successful_files=0,
                failed_files=0,
                error_message=f"Error processing file read request: {str(e)}"
            )
    
    async def _read_single_file(self, file_spec: FileSpec, encoding: str) -> SingleFileResult:
        """Read a single file and return structured result."""
        try:
            file_path = Path(file_spec.file_path)
            
            # Check if file exists
            if not file_path.exists():
                return SingleFileResult(
                    success=False,
                    file_path=str(file_path),
                    error_message=f"File does not exist: {file_path}"
                )
            
            if not file_path.is_file():
                return SingleFileResult(
                    success=False,
                    file_path=str(file_path),
                    error_message=f"Path is not a file: {file_path}"
                )
            
            # Read file content
            try:
                content = file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                # Try with error handling
                content = file_path.read_text(encoding=encoding, errors='replace')
            
            # Get file stats
            stat = file_path.stat()
            lines = content.splitlines()
            
            # Apply line range if specified
            if file_spec.line_range:
                start_line, end_line = file_spec.line_range
                # Convert to 0-based indexing and ensure valid range
                start_idx = max(0, start_line - 1)
                end_idx = min(len(lines), end_line)
                lines = lines[start_idx:end_idx]
                content = '\n'.join(lines)
            
            # Format content with line numbers if focus lines are specified
            if file_spec.focus_lines:
                formatted_lines = []
                line_offset = file_spec.line_range[0] - 1 if file_spec.line_range else 0
                
                for i, line in enumerate(lines, 1 + line_offset):
                    prefix = f"{i:4d}*: " if i in file_spec.focus_lines else f"{i:4d}: "
                    formatted_lines.append(f"{prefix}{line}")
                content = '\n'.join(formatted_lines)
            
            return SingleFileResult(
                success=True,
                file_path=str(file_path),
                content=content,
                line_count=len(lines),
                size_bytes=stat.st_size,
                highlighted_lines=file_spec.focus_lines or []
            )
            
        except Exception as e:
            return SingleFileResult(
                success=False,
                file_path=file_spec.file_path,
                error_message=f"Error reading file: {str(e)}"
            )


# SECURITY WARNING: FileWriteTool is currently DISABLED for security reasons
#
# The FileWriteTool allows LLMs to write arbitrary content to the filesystem,
# which poses significant security risks:
#
# 1. Code Injection: LLMs could write malicious scripts or executables
# 2. Data Corruption: Accidental overwriting of important files
# 3. System Compromise: Writing to system directories or configuration files
# 4. Privilege Escalation: Creating files with elevated permissions
# 5. Backdoor Creation: Writing persistent access mechanisms
#
# Before enabling this tool, implement proper security measures:
# - Sandboxed execution environment
# - File path validation and whitelisting
# - Content scanning and validation
# - User permission checks
# - Audit logging of all write operations
# - Size and type restrictions
#
# Uncomment and modify the class below only after implementing these safeguards.

# class FileWriteTool(BaseTool):
#     """Tool for writing file contents with structured output.
#
#     ⚠️  SECURITY WARNING: This tool is disabled due to security concerns.
#     See comments above for required security measures before enabling.
#     """
#
#     @property
#     def name(self) -> str:
#         return "write_file"
#
#     @property
#     def description(self) -> str:
#         return "Write content to a file, optionally creating parent directories"
#
#     @property
#     def input_schema(self):
#         return FileWriteRequest
#
#     @property
#     def output_schema(self):
#         return FileWriteResponse
#
#     async def execute(
#         self,
#         input_data: FileWriteRequest,
#         context: ToolExecutionContext
#     ) -> FileWriteResponse:
#         """Execute file writing with structured response."""
#         # SECURITY: Return error instead of executing
#         return FileWriteResponse(
#             success=False,
#             file_path=input_data.file_path,
#             error_message="FileWriteTool is disabled for security reasons. See source code comments for details."
#         )
#
#         # Original implementation (commented out for security):
#         # try:
#         #     file_path = Path(input_data.file_path)
#         #     created_dirs = []
#         #
#         #     # Create parent directories if needed
#         #     if input_data.create_dirs and not file_path.parent.exists():
#         #         file_path.parent.mkdir(parents=True, exist_ok=True)
#         #         created_dirs.append(str(file_path.parent))
#         #
#         #     # Write file content
#         #     file_path.write_text(input_data.content, encoding=input_data.encoding)
#         #
#         #     # Get bytes written
#         #     bytes_written = len(input_data.content.encode(input_data.encoding))
#         #
#         #     return FileWriteResponse(
#         #         success=True,
#         #         file_path=str(file_path),
#         #         bytes_written=bytes_written,
#         #         created_dirs=created_dirs
#         #     )
#         #
#         # except Exception as e:
#         #     return FileWriteResponse(
#         #         success=False,
#         #         file_path=input_data.file_path,
#         #         error_message=f"Error writing file: {str(e)}"
#         #     )


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents with structured output."""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "List the contents of a directory with detailed file information"
    
    @property
    def input_schema(self):
        return ListDirectoryRequest
    
    @property
    def output_schema(self):
        return ListDirectoryResponse
    
    async def execute(
        self, 
        input_data: ListDirectoryRequest, 
        context: ToolExecutionContext
    ) -> ListDirectoryResponse:
        """Execute directory listing with structured response."""
        try:
            dir_path = Path(input_data.directory_path)
            
            # Check if directory exists
            if not dir_path.exists():
                return ListDirectoryResponse(
                    success=False,
                    directory_path=str(dir_path),
                    error_message=f"Directory does not exist: {dir_path}"
                )
            
            if not dir_path.is_dir():
                return ListDirectoryResponse(
                    success=False,
                    directory_path=str(dir_path),
                    error_message=f"Path is not a directory: {dir_path}"
                )
            
            items = []
            total_files = 0
            total_directories = 0
            
            # Get directory contents
            if input_data.recursive:
                # Recursive listing
                if input_data.pattern:
                    paths = dir_path.rglob(input_data.pattern)
                else:
                    paths = dir_path.rglob("*")
            else:
                # Non-recursive listing
                if input_data.pattern:
                    paths = dir_path.glob(input_data.pattern)
                else:
                    paths = dir_path.iterdir()
            
            for path in paths:
                # Skip hidden files if not requested
                if not input_data.show_hidden and path.name.startswith('.'):
                    continue
                
                # Skip excluded patterns
                if should_exclude_path(path.parts):
                    continue
                
                try:
                    stat = path.stat()
                    
                    # Get file info
                    file_info = FileInfo(
                        name=path.name,
                        path=str(path),
                        is_directory=path.is_dir(),
                        size_bytes=stat.st_size if path.is_file() else None,
                        modified_time=str(stat.st_mtime),
                        permissions=oct(stat.st_mode)[-3:]
                    )
                    
                    items.append(file_info)
                    
                    if path.is_file():
                        total_files += 1
                    elif path.is_dir():
                        total_directories += 1
                        
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue
            
            # Sort items by name
            items.sort(key=lambda x: (not x.is_directory, x.name.lower()))
            
            return ListDirectoryResponse(
                success=True,
                directory_path=str(dir_path),
                items=items,
                total_items=len(items),
                total_files=total_files,
                total_directories=total_directories
            )
            
        except Exception as e:
            return ListDirectoryResponse(
                success=False,
                directory_path=input_data.directory_path,
                error_message=f"Error listing directory: {str(e)}"
            )