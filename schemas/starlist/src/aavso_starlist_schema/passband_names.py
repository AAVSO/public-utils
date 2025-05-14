from enum import StrEnum

__all__ = ["AAVSOFilters"]

# Restricted list that smart telescopes may report
class AAVSOFilters(StrEnum):
    TG = "TG"
    TR = "TR"
    TB = "TB"
    L3 = "L3"
    L4 = "L4"
