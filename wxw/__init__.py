import warnings
from functools import wraps


def deprecated(func):
    """
    这是一个装饰器，用于标记函数为已弃用。当使用该函数时，会发出警告。

    Args:
        func (function): 被装饰的函数。

    Returns:
        function: 包装后的新函数。
    """

    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # 关闭过滤器
        warnings.warn(
            f"调用已弃用的函数 {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2
        )
        warnings.simplefilter('default', DeprecationWarning)  # 恢复过滤器
        return func(*args, **kwargs)

    return new_func
