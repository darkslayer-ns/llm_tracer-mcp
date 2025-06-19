"""
Search tools with structured Pydantic responses.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional
from pydantic import Field

from ...models.base import BaseFrameworkModel
from ...models.tools import ToolExecutionContext
from ...tool_registry import BaseTool
from ...utils.constants import EXCLUDED_PATTERNS

logger = logging.getLogger(__name__)


# Input Models
class SearchFilesRequest(BaseFrameworkModel):
    """Request model for searching files by name/pattern."""
    pattern: str = Field(..., description="Pattern to search for (supports glob patterns, case-insensitive)")
    directory: str = Field(".", description="Directory to search in")
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to filter by (without dots)")
    recursive: bool = Field(True, description="Whether to search recursively")


class SearchInFilesRequest(BaseFrameworkModel):
    """Request model for searching content within files."""
    search_term: str = Field(..., description="Regex pattern to search for within files (case-insensitive)")
    directory: str = Field(".", description="Directory to search in")
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to search in")
    context_lines: int = Field(2, ge=0, le=10, description="Number of context lines around matches")
    max_results: int = Field(300, ge=1, le=1000, description="Maximum number of results to return")


# Output Models
class FileMatch(BaseFrameworkModel):
    """Information about a matched file."""
    file_path: str = Field(..., description="Path to the matched file")
    file_name: str = Field(..., description="Name of the matched file")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    modified_time: Optional[str] = Field(None, description="Last modified time")


class SearchFilesResponse(BaseFrameworkModel):
    """Structured response for file search."""
    success: bool = Field(..., description="Whether the operation was successful")
    pattern: str = Field(..., description="Pattern that was searched for")
    directory: str = Field(..., description="Directory that was searched")
    matches: List[FileMatch] = Field(default_factory=list, description="List of matching files")
    total_matches: int = Field(0, description="Total number of matches found")
    search_time: Optional[float] = Field(None, description="Time taken for search in seconds")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")


class ContentMatch(BaseFrameworkModel):
    """Information about a content match within a file."""
    file_path: str = Field(..., description="Path to the file containing the match")
    line_number: int = Field(..., description="Line number of the match")
    line_content: str = Field(..., description="Content of the matching line")
    context_before: List[str] = Field(default_factory=list, description="Lines before the match")
    context_after: List[str] = Field(default_factory=list, description="Lines after the match")
    match_start: Optional[int] = Field(None, description="Start position of match in line")
    match_end: Optional[int] = Field(None, description="End position of match in line")


class SearchInFilesResponse(BaseFrameworkModel):
    """Structured response for content search."""
    success: bool = Field(..., description="Whether the operation was successful")
    search_term: str = Field(..., description="Term that was searched for")
    directory: str = Field(..., description="Directory that was searched")
    matches: List[ContentMatch] = Field(default_factory=list, description="List of content matches")
    total_matches: int = Field(0, description="Total number of matches found")
    files_searched: int = Field(0, description="Number of files searched")
    search_time: Optional[float] = Field(None, description="Time taken for search in seconds")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")


# Tool Implementations
class SearchFilesTool(BaseTool):
    """Tool for searching files by name/pattern with structured output."""
    
    @property
    def name(self) -> str:
        return "search_files"
    
    @property
    def description(self) -> str:
        return "Search for files by name or pattern in a directory"
    
    @property
    def input_schema(self):
        return SearchFilesRequest
    
    @property
    def output_schema(self):
        return SearchFilesResponse
    
    async def execute(
        self, 
        input_data: SearchFilesRequest, 
        context: ToolExecutionContext
    ) -> SearchFilesResponse:
        """Execute file search with structured response."""
        import time
        start_time = time.time()
        
        try:
            search_dir = Path(input_data.directory)
            
            # Check if directory exists
            if not search_dir.exists():
                return SearchFilesResponse(
                    success=False,
                    pattern=input_data.pattern,
                    directory=input_data.directory,
                    error_message=f"Directory does not exist: {search_dir}"
                )
            
            if not search_dir.is_dir():
                return SearchFilesResponse(
                    success=False,
                    pattern=input_data.pattern,
                    directory=input_data.directory,
                    error_message=f"Path is not a directory: {search_dir}"
                )
            
            matches = []
            
            # Determine search method
            if input_data.recursive:
                if input_data.file_extensions:
                    # Search with specific extensions
                    for ext in input_data.file_extensions:
                        ext_pattern = f"*.{ext}" if not ext.startswith('.') else f"*{ext}"
                        for file_path in search_dir.rglob(ext_pattern):
                            if file_path.is_file() and self._matches_pattern(
                                file_path.name, input_data.pattern
                            ):
                                matches.append(file_path)
                else:
                    # Search all files
                    for file_path in search_dir.rglob(input_data.pattern):
                        if file_path.is_file():
                            matches.append(file_path)
            else:
                # Non-recursive search
                if input_data.file_extensions:
                    for ext in input_data.file_extensions:
                        ext_pattern = f"*.{ext}" if not ext.startswith('.') else f"*{ext}"
                        for file_path in search_dir.glob(ext_pattern):
                            if file_path.is_file() and self._matches_pattern(
                                file_path.name, input_data.pattern
                            ):
                                matches.append(file_path)
                else:
                    for file_path in search_dir.glob(input_data.pattern):
                        if file_path.is_file():
                            matches.append(file_path)
            
            # Convert to FileMatch objects
            file_matches = []
            for file_path in matches:
                try:
                    stat = file_path.stat()
                    file_match = FileMatch(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        size_bytes=stat.st_size,
                        modified_time=str(stat.st_mtime)
                    )
                    file_matches.append(file_match)
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue
            
            # Sort by file name
            file_matches.sort(key=lambda x: x.file_name.lower())
            
            search_time = time.time() - start_time
            
            return SearchFilesResponse(
                success=True,
                pattern=input_data.pattern,
                directory=input_data.directory,
                matches=file_matches,
                total_matches=len(file_matches),
                search_time=search_time
            )
            
        except Exception as e:
            search_time = time.time() - start_time
            return SearchFilesResponse(
                success=False,
                pattern=input_data.pattern,
                directory=input_data.directory,
                search_time=search_time,
                error_message=f"Error searching files: {str(e)}"
            )
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches the pattern (case-insensitive)."""
        import fnmatch
        
        # Always case-insensitive
        filename = filename.lower()
        pattern = pattern.lower()
        
        return fnmatch.fnmatch(filename, pattern)


class SearchInFilesTool(BaseTool):
    """Tool for searching content within files with structured output."""
    
    @property
    def name(self) -> str:
        return "search_in_files"
    
    @property
    def description(self) -> str:
        return "Search for text content within files and return matches with context"
    
    @property
    def input_schema(self):
        return SearchInFilesRequest
    
    @property
    def output_schema(self):
        return SearchInFilesResponse
    
    async def execute(
        self,
        input_data: SearchInFilesRequest,
        context: ToolExecutionContext
    ) -> SearchInFilesResponse:
        """Execute content search with structured response."""
        start_time = time.time()
        
        try:
            search_dir = Path(input_data.directory)
            
            # Check if directory exists
            if not search_dir.exists():
                return SearchInFilesResponse(
                    success=False,
                    search_term=input_data.search_term,
                    directory=input_data.directory,
                    error_message=f"Directory does not exist: {search_dir}"
                )
            
            # Use grep for case-insensitive regex search
            if self._has_command("grep"):
                return await self._grep_search(input_data, start_time)
            else:
                return SearchInFilesResponse(
                    success=False,
                    search_term=input_data.search_term,
                    directory=input_data.directory,
                    search_time=time.time() - start_time,
                    error_message="grep is not available on this system"
                )
                
        except Exception as e:
            search_time = time.time() - start_time
            return SearchInFilesResponse(
                success=False,
                search_term=input_data.search_term,
                directory=input_data.directory,
                search_time=search_time,
                error_message=f"Error searching in files: {str(e)}"
            )
    
    
    async def _grep_search(self, input_data: SearchInFilesRequest, start_time: float) -> SearchInFilesResponse:
        """Use grep for case-insensitive regex searching with optimized exclusions."""
        cmd = [
            "grep",
            "-r",
            "-n",
            "-i",  # Always case-insensitive
            "-E",  # Always extended regex
            f"-C{input_data.context_lines}"
        ]
        
        # Optimize exclusion patterns - separate files from directories
        file_exclusions = []
        dir_exclusions = []
        
        for pattern in EXCLUDED_PATTERNS:
            if self._is_file_pattern(pattern):
                file_exclusions.append(pattern)
            else:
                dir_exclusions.append(pattern)
        
        # Add all exclusions from constants
        for pattern in dir_exclusions:
            cmd.extend(["--exclude-dir", pattern])
        for pattern in file_exclusions:
            cmd.extend(["--exclude", pattern])
        
        if input_data.file_extensions:
            for ext in input_data.file_extensions:
                # Handle both "py" and ".py" formats
                clean_ext = ext.lstrip('.')
                cmd.extend(["--include", f"*.{clean_ext}"])
        
        # Validate and sanitize the regex pattern
        validated_pattern = self._validate_regex_pattern(input_data.search_term)
        if not validated_pattern:
            search_time = time.time() - start_time
            return SearchInFilesResponse(
                success=False,
                search_term=input_data.search_term,
                directory=input_data.directory,
                search_time=search_time,
                error_message=f"Invalid regex pattern: {input_data.search_term}"
            )
        
        # Add pattern first, then directory
        cmd.append(validated_pattern)
        cmd.append(input_data.directory)
        
        total_exclusions = len(dir_exclusions) + len(file_exclusions)
        logger.info(f"Executing grep search: pattern='{validated_pattern}', directory='{input_data.directory}', extensions={input_data.file_extensions}")
        logger.debug(f"Applied {total_exclusions} exclusion patterns from constants")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Check for regex errors or other failures
            # Note: grep returns 1 when no matches found, 2 for errors
            if result.returncode > 1:
                search_time = time.time() - start_time
                error_message = result.stderr.strip() if result.stderr else f"Command failed with exit code {result.returncode}"
                logger.error(f"Grep search failed: {error_message} (exit code: {result.returncode})")
                return SearchInFilesResponse(
                    success=False,
                    search_term=input_data.search_term,
                    directory=input_data.directory,
                    search_time=search_time,
                    error_message=f"Grep search failed: {error_message}"
                )
            
            matches = self._parse_grep_output(result.stdout)
            
            # Apply result limiting to respect max_results parameter
            limited_matches = matches[:input_data.max_results]
            was_limited = len(matches) > input_data.max_results
            
            # Count unique files that had matches
            unique_files = set(match.file_path for match in limited_matches)
            files_with_matches = len(unique_files)
            
            search_time = time.time() - start_time
            
            if was_limited:
                logger.info(f"Grep search completed: found {len(matches)} total matches, limited to {len(limited_matches)} in {files_with_matches} files (search time: {search_time:.3f}s)")
            else:
                logger.info(f"Grep search completed: found {len(limited_matches)} matches in {files_with_matches} files (search time: {search_time:.3f}s)")
            
            return SearchInFilesResponse(
                success=True,
                search_term=input_data.search_term,
                directory=input_data.directory,
                matches=limited_matches,
                total_matches=len(limited_matches),
                files_searched=files_with_matches,
                search_time=search_time
            )
            
        except subprocess.TimeoutExpired:
            search_time = time.time() - start_time
            return SearchInFilesResponse(
                success=False,
                search_term=input_data.search_term,
                directory=input_data.directory,
                search_time=search_time,
                error_message="Search timed out after 30 seconds"
            )
    
    def _parse_grep_output(self, output: str) -> List[ContentMatch]:
        """Parse grep output into ContentMatch objects."""
        matches = []
        lines = output.strip().split('\n')
        
        logger.debug(f"Parsing grep output: {len(lines)} total lines")
        
        parsed_matches = 0
        skipped_lines = 0
        
        for line in lines:
            if not line or line == '--':
                skipped_lines += 1
                continue
            
            # Skip context lines (they use '-' separator instead of ':')
            # Only process actual match lines (they use ':' separator)
            if ':' in line:
                # Parse format: file:line:content
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        file_path, line_num, content = parts
                        matches.append(ContentMatch(
                            file_path=file_path,
                            line_number=int(line_num),
                            line_content=content,
                            context_before=[],  # Context parsing would be more complex
                            context_after=[]
                        ))
                        parsed_matches += 1
                    except ValueError:
                        skipped_lines += 1
                        continue
            else:
                skipped_lines += 1
        
        logger.debug(f"Parsed {parsed_matches} matches, skipped {skipped_lines} lines (context/separators)")
        
        return matches
    
    def _has_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([command, "--version"], capture_output=True, timeout=5)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _is_file_pattern(self, pattern: str) -> bool:
        """
        Determine if a pattern represents a file or directory.
        
        Args:
            pattern: The exclusion pattern to classify
            
        Returns:
            True if it's a file pattern, False if it's a directory pattern
        """
        # Glob patterns starting with * are always file patterns
        if pattern.startswith('*'):
            return True
        
        # Known file patterns (not directories)
        file_patterns = {
            'Thumbs.db', '.DS_Store'
        }
        if pattern in file_patterns:
            return True
        
        # Patterns with file extensions are likely files
        if pattern.count('.') >= 2:  # e.g., "file.min.js"
            return True
        
        # Everything else is treated as directory pattern
        return False
    
    def _validate_regex_pattern(self, pattern: str) -> str:
        """
        Validate and fix common regex pattern issues for grep.
        
        Args:
            pattern: The regex pattern to validate
            
        Returns:
            Validated pattern or empty string if invalid
        """
        import re
        
        try:
            # Test if the pattern is valid regex
            re.compile(pattern)
            
            # Fix common grep regex issues
            fixed_pattern = pattern
            
            # Count parentheses to ensure they're balanced
            open_parens = fixed_pattern.count('(') - fixed_pattern.count('\\(')
            close_parens = fixed_pattern.count(')') - fixed_pattern.count('\\)')
            
            if open_parens != close_parens:
                logger.warning(f"Unbalanced parentheses in pattern: {pattern}")
                # Try to fix by escaping unmatched parentheses
                if open_parens > close_parens:
                    # More open than close - escape the extras
                    fixed_pattern = re.sub(r'(?<!\\)\(', r'\\(', fixed_pattern)
                elif close_parens > open_parens:
                    # More close than open - escape the extras
                    fixed_pattern = re.sub(r'(?<!\\)\)', r'\\)', fixed_pattern)
            
            # Test the fixed pattern
            re.compile(fixed_pattern)
            
            if fixed_pattern != pattern:
                logger.info(f"Fixed regex pattern: '{pattern}' -> '{fixed_pattern}'")
            
            return fixed_pattern
            
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            # Try to create a simple literal search as fallback
            escaped_pattern = re.escape(pattern)
            logger.info(f"Using literal search fallback: '{escaped_pattern}'")
            return escaped_pattern