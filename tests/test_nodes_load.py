"""
Test that all nodes load correctly

This test verifies that:
1. All 5 nodes can be imported
2. Node registrations are correct
3. All required methods are present
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_node_imports():
    """Test that nodes can be imported"""
    print("Testing node imports...")

    try:
        from nodes import (
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
            Equirect360EmptyLatent,
            Equirect360KSampler,
            Equirect360VAEDecode,
            Equirect360EdgeBlender,
            Equirect360Viewer,
        )
        print("✅ All nodes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import nodes: {e}")
        return False


def test_node_registrations():
    """Test that node registrations are correct"""
    print("\nTesting node registrations...")

    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    expected_nodes = {
        "Equirect360EmptyLatent": "360° Empty Latent",
        "Equirect360KSampler": "360° KSampler",
        "Equirect360VAEDecode": "360° VAE Decode",
        "Equirect360EdgeBlender": "360° Edge Blender",
        "Equirect360Viewer": "360° Viewer",
    }

    # Check all expected nodes are registered
    for node_class, display_name in expected_nodes.items():
        if node_class not in NODE_CLASS_MAPPINGS:
            print(f"❌ {node_class} not in NODE_CLASS_MAPPINGS")
            return False

        if node_class not in NODE_DISPLAY_NAME_MAPPINGS:
            print(f"❌ {node_class} not in NODE_DISPLAY_NAME_MAPPINGS")
            return False

        if NODE_DISPLAY_NAME_MAPPINGS[node_class] != display_name:
            print(f"❌ {node_class} has wrong display name: {NODE_DISPLAY_NAME_MAPPINGS[node_class]}")
            return False

        print(f"✅ {node_class} → {display_name}")

    print(f"\n✅ All {len(expected_nodes)} nodes registered correctly")
    return True


def test_node_structure():
    """Test that nodes have required methods"""
    print("\nTesting node structure...")

    from nodes import NODE_CLASS_MAPPINGS

    required_methods = ["INPUT_TYPES"]
    required_attributes = ["RETURN_TYPES", "FUNCTION", "CATEGORY"]

    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        # Check class methods
        for method in required_methods:
            if not hasattr(node_class, method):
                print(f"❌ {node_name} missing method: {method}")
                return False

        # Check class attributes
        for attr in required_attributes:
            if not hasattr(node_class, attr):
                print(f"❌ {node_name} missing attribute: {attr}")
                return False

        # Check INPUT_TYPES returns a dict
        input_types = node_class.INPUT_TYPES()
        if not isinstance(input_types, dict):
            print(f"❌ {node_name}.INPUT_TYPES() must return a dict")
            return False

        # Check FUNCTION attribute points to an existing method
        if not hasattr(node_class, node_class.FUNCTION):
            print(f"❌ {node_name}.FUNCTION '{node_class.FUNCTION}' does not exist")
            return False

        print(f"✅ {node_name} structure valid")

    print(f"\n✅ All nodes have correct structure")
    return True


def test_utility_imports():
    """Test that utility functions can be imported"""
    print("\nTesting utility imports...")

    try:
        from utils import (
            get_equirect_dimensions,
            validate_aspect_ratio,
            apply_circular_padding,
            remove_circular_padding,
            create_circular_padding_wrapper,
            blend_edges,
            check_edge_continuity,
        )
        print("✅ All utilities imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import utilities: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("ComfyUI-DiT360 Node Loading Tests")
    print("="*60)

    tests = [
        ("Node Imports", test_node_imports),
        ("Node Registrations", test_node_registrations),
        ("Node Structure", test_node_structure),
        ("Utility Imports", test_utility_imports),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Nodes are ready to use.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Fix issues before using.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
