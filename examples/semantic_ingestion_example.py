"""
Example of using the SemanticIngestionTool as an MCP tool.
This demonstrates how to register and use the ingestion tools in an MCP server.
"""

import asyncio
import tempfile
from pathlib import Path

from llm_mcp.server.mcp_server import MCPServer
from llm_mcp.tools.semantic import SemanticIngestionTool, SemanticQueryTool
from llm_mcp.models.semantic_ingestion import (
    SemanticIngestionRequest, SemanticQueryRequest
)
from llm_mcp.models.tools import ToolExecutionContext


async def create_sample_repository(base_path: Path) -> Path:
    """Create a sample repository for testing."""
    repo_path = base_path / "sample_repo"
    repo_path.mkdir(exist_ok=True)
    
    # Create a Python utility module
    utils_py = repo_path / "utils.py"
    utils_py.write_text('''
"""
Utility functions for data processing and validation.
"""

import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def sanitize_input(text: str) -> str:
    """Sanitize user input by removing dangerous characters."""
    # Remove HTML tags and script content
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    return text.strip()


class DataProcessor:
    """Process and transform data structures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0
    
    def process_records(self, records: List[Dict]) -> List[Dict]:
        """Process a list of data records."""
        processed = []
        
        for record in records:
            if self.validate_record(record):
                processed_record = self.transform_record(record)
                processed.append(processed_record)
                self.processed_count += 1
        
        return processed
    
    def validate_record(self, record: Dict) -> bool:
        """Validate a single data record."""
        required_fields = self.config.get('required_fields', [])
        
        for field in required_fields:
            if field not in record or not record[field]:
                return False
        
        return True
    
    def transform_record(self, record: Dict) -> Dict:
        """Transform a data record according to configuration."""
        transformed = record.copy()
        
        # Add timestamp
        transformed['processed_at'] = datetime.utcnow().isoformat()
        
        # Apply transformations
        transformations = self.config.get('transformations', {})
        for field, transform_type in transformations.items():
            if field in transformed:
                if transform_type == 'uppercase':
                    transformed[field] = str(transformed[field]).upper()
                elif transform_type == 'lowercase':
                    transformed[field] = str(transformed[field]).lower()
                elif transform_type == 'sanitize':
                    transformed[field] = sanitize_input(str(transformed[field]))
        
        return transformed


def export_to_json(data: Any, filepath: str) -> bool:
    """Export data to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error exporting to JSON: {e}")
        return False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return {}
''')
    
    # Create a TypeScript service module
    service_ts = repo_path / "service.ts"
    service_ts.write_text('''
/**
 * User service for managing user operations
 */

interface User {
    id: number;
    username: string;
    email: string;
    createdAt: Date;
    isActive: boolean;
}

interface CreateUserRequest {
    username: string;
    email: string;
    password: string;
}

interface UpdateUserRequest {
    username?: string;
    email?: string;
    isActive?: boolean;
}

class UserService {
    private users: Map<number, User> = new Map();
    private nextId: number = 1;

    /**
     * Create a new user
     */
    async createUser(request: CreateUserRequest): Promise<User> {
        // Validate input
        if (!this.isValidEmail(request.email)) {
            throw new Error('Invalid email format');
        }

        if (!this.isValidUsername(request.username)) {
            throw new Error('Invalid username format');
        }

        // Check if user already exists
        if (this.findUserByEmail(request.email)) {
            throw new Error('User with this email already exists');
        }

        if (this.findUserByUsername(request.username)) {
            throw new Error('User with this username already exists');
        }

        // Create user
        const user: User = {
            id: this.nextId++,
            username: request.username,
            email: request.email,
            createdAt: new Date(),
            isActive: true
        };

        this.users.set(user.id, user);
        return user;
    }

    /**
     * Get user by ID
     */
    async getUserById(id: number): Promise<User | null> {
        return this.users.get(id) || null;
    }

    /**
     * Update user information
     */
    async updateUser(id: number, request: UpdateUserRequest): Promise<User | null> {
        const user = this.users.get(id);
        if (!user) {
            return null;
        }

        // Validate email if provided
        if (request.email && !this.isValidEmail(request.email)) {
            throw new Error('Invalid email format');
        }

        // Validate username if provided
        if (request.username && !this.isValidUsername(request.username)) {
            throw new Error('Invalid username format');
        }

        // Update user
        const updatedUser: User = {
            ...user,
            ...request
        };

        this.users.set(id, updatedUser);
        return updatedUser;
    }

    /**
     * Delete user
     */
    async deleteUser(id: number): Promise<boolean> {
        return this.users.delete(id);
    }

    /**
     * Get all active users
     */
    async getActiveUsers(): Promise<User[]> {
        return Array.from(this.users.values()).filter(user => user.isActive);
    }

    /**
     * Search users by username or email
     */
    async searchUsers(query: string): Promise<User[]> {
        const lowercaseQuery = query.toLowerCase();
        return Array.from(this.users.values()).filter(user =>
            user.username.toLowerCase().includes(lowercaseQuery) ||
            user.email.toLowerCase().includes(lowercaseQuery)
        );
    }

    private findUserByEmail(email: string): User | undefined {
        return Array.from(this.users.values()).find(user => user.email === email);
    }

    private findUserByUsername(username: string): User | undefined {
        return Array.from(this.users.values()).find(user => user.username === username);
    }

    private isValidEmail(email: string): boolean {
        const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
        return emailRegex.test(email);
    }

    private isValidUsername(username: string): boolean {
        // Username should be 3-20 characters, alphanumeric and underscores only
        const usernameRegex = /^[a-zA-Z0-9_]{3,20}$/;
        return usernameRegex.test(username);
    }
}

export { UserService, User, CreateUserRequest, UpdateUserRequest };
''')
    
    # Create a README file
    readme_md = repo_path / "README.md"
    readme_md.write_text('''
# Sample Repository

This is a sample repository for testing semantic ingestion capabilities.

## Features

- **Data Processing**: Utilities for processing and validating data
- **User Management**: TypeScript service for managing users
- **Input Validation**: Email and username validation functions
- **Data Export**: JSON export functionality

## Files

- `utils.py` - Python utilities for data processing and validation
- `service.ts` - TypeScript user service with CRUD operations
- `README.md` - This documentation file

## Usage

The utilities in this repository can be used for:

1. **Data Validation**: Validate emails, sanitize input, check data integrity
2. **User Operations**: Create, read, update, delete user records
3. **Data Processing**: Transform and export data in various formats
4. **Configuration Management**: Load and manage application configuration

## Examples

### Python Data Processing

```python
from utils import DataProcessor, validate_email

# Validate email
is_valid = validate_email("user@example.com")

# Process data records
processor = DataProcessor({
    'required_fields': ['name', 'email'],
    'transformations': {'name': 'uppercase'}
})
processed = processor.process_records(records)
```

### TypeScript User Service

```typescript
import { UserService } from './service';

const userService = new UserService();

// Create user
const user = await userService.createUser({
    username: 'johndoe',
    email: 'john@example.com',
    password: 'securepassword'
});

// Search users
const results = await userService.searchUsers('john');
```
''')
    
    return repo_path


async def demonstrate_mcp_ingestion():
    """Demonstrate using semantic ingestion tools via MCP server."""
    print("🚀 Semantic Ingestion MCP Example")
    print("=" * 50)
    
    # Create temporary directory and sample repository
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample repository
        print("📁 Creating sample repository...")
        repo_path = await create_sample_repository(temp_path)
        print(f"   Repository created at: {repo_path}")
        
        # Initialize MCP server with semantic ingestion tools
        print("\n🔧 Setting up MCP server with semantic tools...")
        server = MCPServer("semantic_ingestion_server")
        
        # Register the semantic ingestion tools
        ingestion_tool = SemanticIngestionTool()
        query_tool = SemanticQueryTool()
        
        server.register_tool(ingestion_tool)
        server.register_tool(query_tool)
        
        print(f"   Registered tools: {[tool.name for tool in [ingestion_tool, query_tool]]}")
        
        # Simulate MCP tool execution context
        context = ToolExecutionContext(
            session_id="mcp_demo_session",
            request_id="demo_001"
        )
        
        # Step 1: Ingest the repository
        print("\n📚 Step 1: Ingesting repository via MCP tool...")
        
        ingestion_request = SemanticIngestionRequest(
            repository_path=str(repo_path),
            languages=["python", "typescript", "markdown"],
            embedding_model="codebert_st",
            force_reindex=True
        )
        
        # Execute ingestion tool
        ingestion_response = await server.execute_tool(
            tool_name="semantic_ingestion",
            input_data=ingestion_request,
            context=context
        )
        
        if ingestion_response.success:
            print(f"✅ Repository ingested successfully!")
            print(f"   Database: {ingestion_response.database_path}")
            print(f"   Files processed: {ingestion_response.indexing_stats.files_processed}")
            print(f"   Chunks created: {ingestion_response.indexing_stats.chunks_created}")
            print(f"   Languages: {', '.join(ingestion_response.indexing_stats.languages_found)}")
            print(f"   Ingestion time: {ingestion_response.ingestion_time:.2f}s")
            
            # Step 2: Query the database
            print("\n🔍 Step 2: Querying via MCP tool...")
            
            # Query 1: Data validation functions
            print("\n   Query 1: Data validation functions...")
            validation_query = SemanticQueryRequest(
                query="email validation input sanitization data validation",
                database_path=ingestion_response.database_path,
                max_results=3,
                similarity_threshold=0.3
            )
            
            validation_response = await server.execute_tool(
                tool_name="semantic_query",
                input_data=validation_query,
                context=context
            )
            
            if validation_response.success:
                print(f"   ✅ Found {validation_response.total_results} validation-related results")
                for i, chunk in enumerate(validation_response.results, 1):
                    print(f"      {i}. {Path(chunk.file_path).name} (lines {chunk.start_line}-{chunk.end_line}) - Score: {chunk.similarity_score:.3f}")
            
            # Query 2: User management operations
            print("\n   Query 2: User management operations...")
            user_query = SemanticQueryRequest(
                query="user management CRUD operations create update delete",
                database_path=ingestion_response.database_path,
                languages=["typescript"],
                max_results=3,
                similarity_threshold=0.3
            )
            
            user_response = await server.execute_tool(
                tool_name="semantic_query",
                input_data=user_query,
                context=context
            )
            
            if user_response.success:
                print(f"   ✅ Found {user_response.total_results} user management results")
                for i, chunk in enumerate(user_response.results, 1):
                    print(f"      {i}. {Path(chunk.file_path).name} (lines {chunk.start_line}-{chunk.end_line}) - Score: {chunk.similarity_score:.3f}")
            
            # Query 3: Configuration and data processing
            print("\n   Query 3: Configuration and data processing...")
            config_query = SemanticQueryRequest(
                query="configuration management data processing transformation",
                database_path=ingestion_response.database_path,
                max_results=3,
                similarity_threshold=0.3
            )
            
            config_response = await server.execute_tool(
                tool_name="semantic_query",
                input_data=config_query,
                context=context
            )
            
            if config_response.success:
                print(f"   ✅ Found {config_response.total_results} configuration/processing results")
                for i, chunk in enumerate(config_response.results, 1):
                    print(f"      {i}. {Path(chunk.file_path).name} (lines {chunk.start_line}-{chunk.end_line}) - Score: {chunk.similarity_score:.3f}")
            
            print(f"\n💾 Persistent database created at: {ingestion_response.database_path}")
            print("   This database can be used later for semantic queries!")
            
        else:
            print(f"❌ Repository ingestion failed: {ingestion_response.error_message}")
        
        print("\n✅ MCP semantic ingestion demonstration completed!")


if __name__ == "__main__":
    print("🧪 Semantic Ingestion MCP Tool Example")
    print("This demonstrates how to use semantic ingestion tools via MCP server")
    print()
    
    try:
        asyncio.run(demonstrate_mcp_ingestion())
        print("\n🎉 Example completed successfully!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install semantic search dependencies with:")
        print("  pip install -e .[semantic]")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()