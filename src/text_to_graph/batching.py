"""Preserve the input position of each provider result or exception."""


def isolated(items, run):
    if not items:
        return []
    try:
        results = list(run(items))
        if len(results) != len(items):
            raise RuntimeError("Model returned the wrong number of results")
        return results
    except Exception as exc:  # noqa: BLE001 - provider failures can have backend-specific types
        if len(items) == 1:
            return [exc.with_traceback(None)]
    middle = len(items) // 2
    return isolated(items[:middle], run) + isolated(items[middle:], run)
