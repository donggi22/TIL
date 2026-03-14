# =============================================================================
# step23.py부터 step32.py까지는 simple_core를 이용해야 합니다.
# is_simple_core = True # step32.py까지
is_simple_core = False # step33.py부터
# =============================================================================

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

# else:
    from dezero.core import Variable
    # from dezero.core import Parameter
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    # from dezero.core import test_mode
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable

setup_variable()