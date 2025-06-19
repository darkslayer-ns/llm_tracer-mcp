import asyncio
from pydantic import BaseModel, Field
from llm_mcp import analyze_repository_with_llm


# Example Pydantic model for structured response
class CodeAnalysis(BaseModel):
    """Example model for code analysis results."""
    summary: str = Field(..., description="Brief summary of the code")
    files_analyzed: list[str] = Field(..., description="List of files that were analyzed")
    key_findings: list[str] = Field(..., description="Key findings from the analysis")
    recommendations: list[str] = Field(..., description="Recommendations for improvement")
    semantic_insights: list[str] = Field(default_factory=list, description="Insights from semantic search")


# Additional analysis model for testing database reuse
class PerformanceAnalysis(BaseModel):
    """Model for performance analysis results."""
    summary: str = Field(..., description="Performance analysis summary")
    files_analyzed: list[str] = Field(..., description="List of files analyzed")
    performance_issues: list[str] = Field(..., description="Performance issues found")
    optimization_suggestions: list[str] = Field(..., description="Optimization suggestions")
    async_patterns: list[str] = Field(default_factory=list, description="Async/await patterns found")


async def test_llm_resp():
    print("🚀 Testing LLMResp with Semantic Search Enhanced Analysis")
    print("🔄 Demonstrating Database Reuse for Multiple Analyses")
    print("=" * 60)
    
    try:
        # Analysis 1: Security Analysis (creates database)
        print("\n🔒 Analysis 1: Security Analysis")
        print("-" * 40)
        
        security_prompt = """"""
        security_message = """You are a security-focused code analyst. Identify any HTTP or GraphQL entrypoints and analyze any vulnerabilities from entrypoint to the final sink.."""

        # First analysis - creates database and returns it for reuse
        security_result, semantic_db = await analyze_repository_with_llm(
            system_prompt=security_prompt,
            user_message=security_message,
            response_model=CodeAnalysis,
            repository_path=".",
            display=True,
            return_db=True  # Return database for reuse
        )
        
        print("\n" + "=" * 60)
        print("📋 SECURITY ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 Security Analysis Data:")
        print(f"   Summary: {security_result.summary[:100]}...")
        print(f"   Files Analyzed: {security_result.files_analyzed}")
        print(f"   Key Findings: {len(security_result.key_findings)} items")
        print(f"   Recommendations: {len(security_result.recommendations)} items")
        print(f"   Semantic Insights: {len(security_result.semantic_insights)} items")
        
        # Analysis 2: Performance Analysis (reuses database)
        print("\n\n⚡ Analysis 2: Performance Analysis")
        print("-" * 40)
        print("📊 Reusing existing semantic database...")
        
        performance_prompt = """You are a performance optimization expert. Analyze the codebase for performance issues and optimization opportunities.

{json_schema}"""

        performance_message = """Please analyze the Python files in this directory with focus on performance optimization.

Look for:
1. Async/await usage patterns
2. Database query efficiency
3. Loop optimizations
4. Memory usage patterns
5. I/O operations efficiency

Provide performance insights and optimization suggestions."""

        # Second analysis - reuses existing database
        performance_result = await analyze_repository_with_llm(
            system_prompt=performance_prompt,
            user_message=performance_message,
            response_model=PerformanceAnalysis,
            semantic_db=semantic_db,  # Reuse the existing database
            display=True
        )
        
        print("\n" + "=" * 60)
        print("📋 PERFORMANCE ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 Performance Analysis Data:")
        print(f"   Summary: {performance_result.summary[:100]}...")
        print(f"   Files Analyzed: {performance_result.files_analyzed}")
        print(f"   Performance Issues: {len(performance_result.performance_issues)} items")
        print(f"   Optimization Suggestions: {len(performance_result.optimization_suggestions)} items")
        print(f"   Async Patterns: {len(performance_result.async_patterns)} items")
        
        # Clean up the database manually
        print("\n🧹 Cleaning up semantic database...")
        await semantic_db.cleanup()
        print("✅ Database cleaned up successfully")
        
        # Final summary
        print("\n" + "=" * 60)
        print("📋 MULTI-ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\n🔒 Security Analysis:")
        print(f"   Files: {len(security_result.files_analyzed)}")
        print(f"   Findings: {len(security_result.key_findings)}")
        print(f"   Recommendations: {len(security_result.recommendations)}")
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   Files: {len(performance_result.files_analyzed)}")
        print(f"   Issues: {len(performance_result.performance_issues)}")
        print(f"   Suggestions: {len(performance_result.optimization_suggestions)}")
        
        print(f"\n💡 Database Reuse Benefits:")
        print(f"   ✅ Embeddings created only once")
        print(f"   ✅ Faster second analysis")
        print(f"   ✅ Consistent semantic understanding")
        print(f"   ✅ Manual cleanup control")
        
        print(f"\n✅ Multi-analysis test completed successfully!")
        
        return {
            'security': security_result,
            'performance': performance_result
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        # Try to cleanup database if it exists
        try:
            if 'semantic_db' in locals():
                await semantic_db.cleanup()
                print("🧹 Emergency cleanup completed")
        except:
            pass
        raise


if __name__ == "__main__":
    asyncio.run(test_llm_resp())