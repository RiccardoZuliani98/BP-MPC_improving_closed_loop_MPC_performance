import sys
import os
import casadi as ca
import pytest

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.symbolic_var import SymbolicVar

def test_check_and_convert_scalar():
    # Test with a scalar value
    value = 5
    expected_shape = 1
    result = SymbolicVar._check_and_convert(value, expected_shape)
    assert isinstance(result, ca.DM)
    assert result.shape == (1, 1)
    assert result[0] == 5

def test_check_and_convert_vector():
    # Test with a vector
    value = [ca.DM.ones(3,1), 2*ca.DM.ones(3,1), 3*ca.DM.ones(3,1)]
    expected_shape = 3
    result = SymbolicVar._check_and_convert(value, expected_shape)
    assert isinstance(result, list) and len(result) == 3, 'Length is wrong'
    assert all([elem.shape == (3, 1) for elem in result]), 'Shape is wrong'
    assert ca.mmin(result[0] == ca.DM.ones(3,1)) == 1, 'Element 1 is wrong'
    assert ca.mmin(result[1] == 2*ca.DM.ones(3,1)) == 1, 'Element 2 is wrong'
    assert ca.mmin(result[2] == 3*ca.DM.ones(3,1)) == 1, 'Element 3 is wrong'

def test_check_and_convert_nested_list():
    # Test with a nested list
    value = [[ca.DM.ones(3,1)], [2*ca.DM.ones(3,1)], [3*ca.DM.ones(3,1)]]
    expected_shape = 3
    result = SymbolicVar._check_and_convert(value, expected_shape)
    assert isinstance(result, list) and len(result) == 3, 'Length is wrong'
    assert all([elem.shape == (3, 1) for elem in result]), 'Shape is wrong'
    assert ca.mmin(result[0] == ca.DM.ones(3,1)) == 1, 'Element 1 is wrong'
    assert ca.mmin(result[1] == 2*ca.DM.ones(3,1)) == 1, 'Element 2 is wrong'
    assert ca.mmin(result[2] == 3*ca.DM.ones(3,1)) == 1, 'Element 3 is wrong'

def test_check_and_convert_matrix():
    # Test with a matrix
    value = [[ca.DM.ones(3,1),ca.DM.ones(3,1),ca.DM.ones(3,1)], [2*ca.DM.ones(3,1),2*ca.DM.ones(3,1),2*ca.DM.ones(3,1)]]
    expected_shape = 3
    result = SymbolicVar._check_and_convert(value, expected_shape)
    assert isinstance(result, list) and len(result) == 2, 'Length of outer list is wrong'
    assert all([len(elem) == 3 for elem in result]), 'Length of inner list is wrong'
    for elem in result:
        assert all([elem_inner.shape[0] == expected_shape for elem_inner in elem]), 'Shape of elements is wrong'

def test_check_and_convert_invalid_shape():
    # Test with an invalid shape
    value = [[ca.DM.ones(3,1)], [2*ca.DM.ones(3,1)], [3*ca.DM.ones(3,1)]]
    expected_shape = 2
    with pytest.raises(AssertionError, match="Dimension of initialization does not match dimension of symbolic variable"):
        SymbolicVar._check_and_convert(value, expected_shape)

def test_set_init_valid_data():
    # Test with valid data
    symbolic_var = SymbolicVar()
    symbolic_var.add_var("x", ca.SX.sym("x", 3, 1))
    symbolic_var.add_var("y", ca.SX.sym("y", 2, 2))
    
    init_data = {
        "x": ca.DM([1, 2, 3]),
        "y": ca.DM([[1, 2], [3, 4]])
    }
    
    symbolic_var.set_init(init_data)
    
    assert "x" in symbolic_var.init
    assert "y" in symbolic_var.init
    assert ca.mmin(symbolic_var.init["x"] == ca.DM([1, 2, 3])) == 1
    assert ca.mmin(symbolic_var.init["y"] == ca.DM([[1, 2], [3, 4]])) == 1

def test_set_init_nonexistent_variable():
    # Test with a variable that does not exist
    symbolic_var = SymbolicVar()
    symbolic_var.add_var("x", ca.SX.sym("x", 3, 1))
    
    init_data = {
        "y": ca.DM([1, 2, 3])
    }
    
    with pytest.raises(AssertionError, match="Cannot initialize variable that does not exist"):
        symbolic_var.set_init(init_data)

def test_set_init_invalid_dimension():
    # Test with invalid dimensions
    symbolic_var = SymbolicVar()
    symbolic_var.add_var("x", ca.SX.sym("x", 3, 1))
    
    init_data = {
        "x": ca.DM([1, 2])  # Incorrect dimension
    }
    
    with pytest.raises(AssertionError, match="Dimension of initialization does not match dimension of symbolic variable"):
        symbolic_var.set_init(init_data)

def test_set_init_nested_list():
    # Test with a nested list
    symbolic_var = SymbolicVar()
    symbolic_var.add_var("x", ca.SX.sym("x", 3, 1))
    
    init_data = {
        "x": [[ca.DM.ones(3,1)], [2*ca.DM.ones(3,1)], [3*ca.DM.ones(3,1)]]
    }
    
    symbolic_var.set_init(init_data)
    
    assert "x" in symbolic_var.init
    assert all([ca.mmin(elem1 == elem2)==1 for elem1,elem2 in zip(symbolic_var.init["x"],[ca.DM.ones(3,1), 2*ca.DM.ones(3,1), 3*ca.DM.ones(3,1)])])

def test_set_init_single_element_list():
    # Test with a single-element list
    symbolic_var = SymbolicVar()
    symbolic_var.add_var("x", ca.SX.sym("x", 3, 1))
    
    init_data = {
        "x": [ca.DM([1, 2, 3])]
    }
    
    symbolic_var.set_init(init_data)
    
    assert "x" in symbolic_var.init
    assert ca.mmin(symbolic_var.init["x"] == ca.DM([1, 2, 3])) == 1



