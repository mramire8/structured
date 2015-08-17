__author__ = 'mramire8'


def unit_cost(query, cost_model=None):
    return query.bow[0]


def intra_cost(query, cost_model=None):
    if cost_model is None:
        raise ValueError("Cost model is not available.")
    c = 0
    x_text = query.snippet
    if x_text is not None:
        c = [_cost_intrapolated(len(x), cost_model.values(), cost_model.keys()) for x in x_text]

    return sum(c)


def _cost_intrapolated(x, cost, kvalues):
    import numpy as np

    binx = min(np.digitize([x], kvalues)[0], len(cost)-1)
    lbbinx = max(binx-1, 0)

    y1 = cost[lbbinx] if lbbinx>=0  else 0
    y2 = cost[binx]
    x1 = kvalues[lbbinx] if lbbinx >=0 else 0
    x2 = kvalues[binx]

    m = (y2-y1) / (x2-x1)
    b = y2 - m * x2

    if x < kvalues[0]:
        y = cost[0]
    elif x > kvalues[-1]:
        y = cost[-1]
    else:
        y = (m * x) + b
    return y