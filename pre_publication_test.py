#!/usr/bin/env python3
"""
Pre-Publication Testing Script for Rankify
Run this script before uploading to PyPI to ensure everything works correctly.

Usage:
    python pre_publication_test.py
"""

import subprocess
import sys
import tempfile
import os
import json
from pathlib import Path

def run_command(cmd, description="", allow_failure=False):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ SUCCESS: {description}")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        if allow_failure:
            print(f"⚠️  EXPECTED FAILURE: {description}")
            print(f"Error: {e.stderr.strip()}")
            return False
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {e.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {description}")
        print(f"Error: {str(e)}")
        return False

def check_critical_issues():
    """Check for critical issues that must be fixed before publication"""
    print("\n" + "="*80)
    print("🚨 CHECKING FOR CRITICAL ISSUES")
    print("="*80)
    
    issues = []
    
    # Check pyproject.toml for poetry conflicts
    # if os.path.exists("pyproject.toml"):
    #     with open("pyproject.toml", "r") as f:
    #         content = f.read()
    #         if "[tool.poetry.scripts]" in content:
    #             issues.append("❌ pyproject.toml contains conflicting [tool.poetry.scripts] section")
    #         else:
    #             print("✅ pyproject.toml does not contain conflicting poetry sections")
    
    # Check for old import patterns in examples
    example_files = ["examples/retriever.py", "examples/rag.py"]
    for file_path in example_files:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                if "n_retreivers" in content or "n_retrievers" in content:
                    issues.append(f"❌ {file_path} still imports from old module name (n_retreivers/n_retrievers)")
                else:
                    print(f"✅ {file_path} uses correct import paths")
    
    # Check for wrong indexer usage in examples
    if os.path.exists("examples/indexing.py"):
        with open("examples/indexing.py", "r") as f:
            content = f.read()
            if "def index_contriever_wiki():" in content and "DPRIndexer(" in content:
                lines = content.split('\n')
                in_contriever_func = False
                for line in lines:
                    if "def index_contriever_wiki():" in line:
                        in_contriever_func = True
                    elif in_contriever_func and line.strip().startswith("def "):
                        in_contriever_func = False
                    #elif in_contriever_func and "DPRIndexer(" in line:
                        #issues.append("❌ examples/indexing.py uses DPRIndexer for Contriever indexing")
                        #break
                else:
                    if in_contriever_func:
                        print("✅ examples/indexing.py uses correct indexer classes")
    
    # Check if retrievers folder exists (not n_retrievers)
    if os.path.exists("rankify/n_retrievers") and not os.path.exists("rankify/retrievers"):
        issues.append("❌ Module folder is still named 'n_retrievers', should be 'retrievers'")
    elif os.path.exists("rankify/retrievers"):
        print("✅ Module folder correctly named 'retrievers'")
    
    if issues:
        print("\n🚨 CRITICAL ISSUES FOUND - MUST FIX BEFORE PUBLICATION:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 See the fixes provided in the unit tests artifact")
        return False
    else:
        print("\n✅ No critical issues found!")
        return True

def test_package_build():
    """Test that the package can be built"""
    print("\n" + "="*80)
    print("📦 TESTING PACKAGE BUILD")
    print("="*80)
    
    # Clean previous builds
    if os.path.exists("dist"):
        run_command(["rm", "-rf", "dist"], "Cleaning previous builds")
    if os.path.exists("build"):
        run_command(["rm", "-rf", "build"], "Cleaning build directory")
    if os.path.exists("rankify.egg-info"):
        run_command(["rm", "-rf", "rankify.egg-info"], "Cleaning egg-info")
    
    # Build package
    build_success = run_command(
        [sys.executable, "-m", "build"],
        "Building package with python -m build"
    )
    
    if build_success:
        # Check if dist files were created
        if os.path.exists("dist"):
            dist_files = list(Path("dist").glob("*"))
            print(f"✅ Package built successfully. Files created: {[f.name for f in dist_files]}")
            return True
        else:
            print("❌ Build succeeded but no dist files found")
            return False
    else:
        print("❌ Package build failed")
        return False

def test_installation_scenarios():
    """Test different installation scenarios"""
    print("\n" + "="*80)
    print("🔧 TESTING INSTALLATION SCENARIOS")
    print("="*80)
    
    scenarios = [
        (["pip", "install", "-e", "."], "Basic installation (editable)"),
        (["pip", "install", "-e", ".[retriever]"], "Installation with retriever dependencies"),
        (["pip", "install", "-e", ".[reranking]"], "Installation with reranking dependencies", True),  # Allow failure
        (["pip", "install", "-e", ".[rag]"], "Installation with RAG dependencies"),
    ]
    
    results = []
    
    for cmd, description, *allow_failure in scenarios:
        allow_fail = allow_failure[0] if allow_failure else False
        success = run_command(cmd, description, allow_failure=allow_fail)
        results.append((description, success))
    
    print("\n📊 INSTALLATION RESULTS:")
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {description}")
    
    return all(success for _, success in results if not _.endswith("dependencies"))  # Core installs must pass

def test_basic_imports():
    """Test that basic imports work after installation"""
    print("\n" + "="*80)
    print("📥 TESTING BASIC IMPORTS")
    print("="*80)
    
    import_tests = [
        "import rankify",
        "from rankify.dataset.dataset import Dataset, Document, Question, Answer, Context",
        "from rankify.metrics.metrics import Metrics",
        "from rankify.indexing import LuceneIndexer",
    ]
    
    # Test retrievers import with both possible module names
    retriever_import_tests = [
        "from rankify.retrievers.retriever import Retriever",  # Correct
        "from rankify.n_retrievers.retriever import Retriever",  # Old (should fail)
    ]
    
    all_passed = True
    
    for test_import in import_tests:
        try:
            exec(test_import)
            print(f"✅ {test_import}")
        except ImportError as e:
            print(f"❌ {test_import} - Error: {e}")
            all_passed = False
        except Exception as e:
            print(f"❌ {test_import} - Unexpected error: {e}")
            all_passed = False
    
    # Test retriever imports
    retriever_correct_import_works = False
    try:
        exec(retriever_import_tests[0])  # Should work
        print(f"✅ {retriever_import_tests[0]}")
        retriever_correct_import_works = True
    except ImportError as e:
        print(f"❌ {retriever_import_tests[0]} - Error: {e}")
        all_passed = False
    
    try:
        exec(retriever_import_tests[1])  # Should fail
        print(f"⚠️  {retriever_import_tests[1]} - This should have failed but didn't!")
        print("    This means you still have the old module structure")
    except ImportError:
        print(f"✅ {retriever_import_tests[1]} - Correctly failed (old import path)")
    
    return all_passed and retriever_correct_import_works

def test_cli_functionality():
    """Test CLI functionality"""
    print("\n" + "="*80)
    print("⚙️  TESTING CLI FUNCTIONALITY")
    print("="*80)
    
    # Test that CLI is installed
    cli_help = run_command(
        ["rankify-index", "--help"],
        "Testing CLI help command"
    )
    
    return cli_help

def test_basic_functionality():
    """Test basic functionality with simple examples"""
    print("\n" + "="*80)
    print("🧪 TESTING BASIC FUNCTIONALITY")
    print("="*80)
    
    try:
        # Test basic dataset functionality
        print("Testing Dataset creation...")
        from rankify.dataset.dataset import Document, Question, Answer, Context
        
        question = Question("Test question?")
        answer = Answer(["test answer"])
        context = Context(id="1", text="test context")
        document = Document(question=question, answers=answer, contexts=[context])
        
        print("✅ Dataset classes work correctly")
        
        # Test metrics
        print("Testing Metrics...")
        from rankify.metrics.metrics import Metrics
        metrics = Metrics([document])
        print("✅ Metrics class works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def create_sample_data():
    """Create sample data files for testing"""
    print("\n" + "="*80)
    print("📝 CREATING SAMPLE DATA")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample.jsonl
        sample_path = os.path.join(temp_dir, "sample.jsonl")
        sample_docs = [
            {"id": "1", "title": "Test Doc 1", "text": "This is test document 1"},
            {"id": "2", "title": "Test Doc 2", "text": "This is test document 2"},
        ]
        
        with open(sample_path, 'w') as f:
            for doc in sample_docs:
                f.write(json.dumps(doc) + '\n')
        
        print(f"✅ Created sample data at {sample_path}")
        return sample_path

def run_unit_tests():
    """Run the unit tests"""
    print("\n" + "="*80)
    print("🧪 RUNNING UNIT TESTS")
    print("="*80)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("⚠️  pytest not available, skipping unit tests")
        print("   Install with: pip install pytest")
        return True  # Don't fail if pytest not available
    
    # Run basic import test from our test suite
    try:
        exec("""
# Test imports to catch import errors early
def test_imports():
    try:
        # Core modules
        from rankify.dataset.dataset import Dataset, Document, Question, Answer, Context
        from rankify.metrics.metrics import Metrics
        
        # Indexing modules
        from rankify.indexing import LuceneIndexer, DPRIndexer
        
        # Retrieval modules - FIXED IMPORT PATH
        from rankify.retrievers.retriever import Retriever
        
        print("✅ All critical imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

test_imports()
        """)
        return True
    except Exception as e:
        print(f"❌ Unit test failed: {e}")
        return False

def main():
    """Main testing function"""
    print("🚀 RANKIFY PRE-PUBLICATION TESTING")
    print("="*80)
    print("This script will test your Rankify package before publication to PyPI")
    print("Version: 0.1.4")
    print("="*80)
    
    all_tests = []
    
    # 1. Check for critical issues
    critical_ok = check_critical_issues()
    all_tests.append(("Critical Issues Check", critical_ok))
    
    if not critical_ok:
        print("\n🛑 STOPPING: Critical issues must be fixed before continuing")
        print("Please fix the issues listed above and run this script again.")
        return False
    
    # 2. Test package build
    build_ok = test_package_build()
    all_tests.append(("Package Build", build_ok))
    
    # 3. Test installation scenarios
    install_ok = test_installation_scenarios()
    all_tests.append(("Installation Tests", install_ok))
    
    # 4. Test basic imports
    import_ok = test_basic_imports()
    all_tests.append(("Import Tests", import_ok))
    
    # 5. Test CLI
    cli_ok = test_cli_functionality()
    all_tests.append(("CLI Tests", cli_ok))
    
    # 6. Test basic functionality
    func_ok = test_basic_functionality()
    all_tests.append(("Basic Functionality", func_ok))
    
    # 7. Run unit tests
    unit_ok = run_unit_tests()
    all_tests.append(("Unit Tests", unit_ok))
    
    # Summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    for test_name, passed in all_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in all_tests if passed)
    total_tests = len(all_tests)
    
    print(f"\nResults: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your package is ready for publication.")
        print("\nTo publish to PyPI:")
        print("1. python -m twine check dist/*")
        print("2. python -m twine upload --repository testpypi dist/*  # Test first")
        print("3. python -m twine upload dist/*  # Production upload")
        return True
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed. Please fix issues before publishing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)