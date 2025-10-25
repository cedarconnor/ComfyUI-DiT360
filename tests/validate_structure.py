"""
Validate ComfyUI-DiT360 project structure

This script checks that all required files exist and have correct structure
WITHOUT importing modules (so it works outside ComfyUI).

Run with: python tests/validate_structure.py
"""

import sys
from pathlib import Path


def validate_file_structure():
    """Check that all required files exist"""
    print("\n" + "="*60)
    print("VALIDATION: File Structure")
    print("="*60)

    root = Path(__file__).parent.parent
    required_files = [
        # Core files
        "__init__.py",
        "nodes.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        ".gitignore",
        # Model implementation
        "dit360/__init__.py",
        "dit360/model.py",
        "dit360/vae.py",
        "dit360/conditioning.py",
        # Utilities
        "utils/__init__.py",
        "utils/equirect.py",
        "utils/padding.py",
        # Tests
        "tests/test_utils.py",
        "tests/test_model_loading.py",
        # Documentation
        "IMPLEMENTATION_STATUS.md",
        "PHASE1_COMPLETE.md",
        "TECHNICAL_DESIGN.md",
        "AGENTS.md",
    ]

    missing = []
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            missing.append(file_path)

    if missing:
        print(f"\n[FAIL] {len(missing)} files missing")
        return False
    else:
        print(f"\n[PASS] All {len(required_files)} required files present")
        return True


def validate_file_contents():
    """Check that key files have expected content"""
    print("\n" + "="*60)
    print("VALIDATION: File Contents")
    print("="*60)

    root = Path(__file__).parent.parent

    checks = [
        # Check __init__.py has NODE_CLASS_MAPPINGS
        ("__init__.py", "NODE_CLASS_MAPPINGS"),
        # Check nodes.py has all 6 nodes
        ("nodes.py", "class DiT360Loader"),
        ("nodes.py", "class DiT360TextEncode"),
        ("nodes.py", "class DiT360Sampler"),
        ("nodes.py", "class DiT360Decode"),
        ("nodes.py", "class Equirect360Process"),
        ("nodes.py", "class Equirect360Preview"),
        # Check model files have key functions
        ("dit360/model.py", "def load_dit360_model"),
        ("dit360/vae.py", "def load_vae"),
        ("dit360/conditioning.py", "def load_t5_encoder"),
        # Check utils have key functions
        ("utils/equirect.py", "def validate_aspect_ratio"),
        ("utils/padding.py", "def apply_circular_padding"),
    ]

    failed = []
    for file_path, expected_content in checks:
        full_path = root / file_path
        if not full_path.exists():
            print(f"  [SKIP] {file_path} (file not found)")
            failed.append((file_path, expected_content))
            continue

        content = full_path.read_text(encoding='utf-8')
        if expected_content in content:
            print(f"  [OK] {file_path} contains '{expected_content}'")
        else:
            print(f"  [FAIL] {file_path} missing '{expected_content}'")
            failed.append((file_path, expected_content))

    if failed:
        print(f"\n[FAIL] {len(failed)} content checks failed")
        return False
    else:
        print(f"\n[PASS] All {len(checks)} content checks passed")
        return True


def count_lines_of_code():
    """Count total lines of code"""
    print("\n" + "="*60)
    print("STATISTICS: Lines of Code")
    print("="*60)

    root = Path(__file__).parent.parent

    python_files = list(root.glob("**/*.py"))
    # Exclude test files and __pycache__
    python_files = [f for f in python_files if "__pycache__" not in str(f)]

    total_lines = 0
    file_count = 0

    for py_file in sorted(python_files):
        try:
            lines = len(py_file.read_text(encoding='utf-8').splitlines())
            relative = py_file.relative_to(root)
            print(f"  {relative}: {lines} lines")
            total_lines += lines
            file_count += 1
        except Exception as e:
            print(f"  [ERROR] {py_file}: {e}")

    print(f"\nTotal: {total_lines} lines across {file_count} Python files")
    return True


def validate_phase2_implementation():
    """Check that Phase 2 components are present"""
    print("\n" + "="*60)
    print("VALIDATION: Phase 2 Implementation")
    print("="*60)

    root = Path(__file__).parent.parent

    phase2_checks = [
        # Model loading
        ("dit360/model.py", "class DiT360Model"),
        ("dit360/model.py", "class DiT360Wrapper"),
        ("dit360/model.py", "def download_dit360_from_huggingface"),
        # VAE
        ("dit360/vae.py", "class DiT360VAE"),
        ("dit360/vae.py", "def download_vae_from_huggingface"),
        # Text encoder
        ("dit360/conditioning.py", "class T5TextEncoder"),
        ("dit360/conditioning.py", "def download_t5_from_huggingface"),
        # Updated nodes
        ("nodes.py", "from .dit360 import load_dit360_model"),
        ("nodes.py", "from .dit360 import text_preprocessing"),
    ]

    failed = []
    for file_path, expected in phase2_checks:
        full_path = root / file_path
        if not full_path.exists():
            print(f"  [FAIL] {file_path} not found")
            failed.append((file_path, expected))
            continue

        content = full_path.read_text(encoding='utf-8')
        if expected in content:
            print(f"  [OK] {expected}")
        else:
            print(f"  [FAIL] {file_path} missing '{expected}'")
            failed.append((file_path, expected))

    if failed:
        print(f"\n[FAIL] {len(failed)} Phase 2 checks failed")
        return False
    else:
        print(f"\n[PASS] All {len(phase2_checks)} Phase 2 components present")
        return True


def run_all_validations():
    """Run all validation checks"""
    print("\n" + "="*60)
    print("ComfyUI-DiT360 Structure Validation")
    print("="*60)

    checks = [
        validate_file_structure,
        validate_file_contents,
        validate_phase2_implementation,
        count_lines_of_code,
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Validation failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "="*60)
    if all(results):
        print("[SUCCESS] All validations passed!")
        print("="*60 + "\n")
        return True
    else:
        print(f"[FAIL] {len(results) - sum(results)} validation(s) failed")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1)
