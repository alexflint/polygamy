import ast


def extract_symbols(module):
    return {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}


def extract_variable_names(*exprs):
    return set.union(*[extract_symbols(ast.parse(expr.strip())) for expr in exprs])


def function_from_expression(expr, varnames=None):
    if varnames is None:
        varnames = extract_variable_names([expr])
    source = 'def f(%s): return %s' % (','.join(varnames), expr)
    code = compile(source, '<polynomial>', mode='exec')
    namespace = {}
    exec(code, namespace)
    return namespace['f']
