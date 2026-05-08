# core/cell.py — base cell class (will be expanded later for immune cells)

class BaseCell:
    """Base class for all cell agents in the simulation.
    Immune cells (TAMs, T cells) will inherit from this in Month 4."""

    def __init__(self, x, y, cell_type="base"):
        self.x = x
        self.y = y
        self.cell_type = cell_type
        self.alive = True
        self.age = 0.0

    def update(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement update()")