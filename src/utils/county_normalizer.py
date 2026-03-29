"""
County name normalization utilities.

Handles inconsistencies between folder names, PDF names, and canonical county names.
"""

import re
from typing import Optional


def normalize_county_name(name: str) -> str:
    """
    Normalize a county name to canonical form: "Xxx County"
    
    Examples:
        "Alachua County Plan" -> "Alachua County"
        "alachua" -> "Alachua County"
        "MIAMI-DADE COUNTY" -> "Miami-Dade County"
        "St. Johns County Plan" -> "St. Johns County"
    """
    if not name:
        return ""
    
    # Clean up
    name = name.strip()
    
    # Remove common suffixes
    suffixes_to_remove = [
        r"\s+comp(rehensive)?\s+plan.*$",
        r"\s+plan.*$",
        r"\s+comprehensive.*$",
        r"\s+\d{4}.*$",  # Year suffixes
    ]
    
    for pattern in suffixes_to_remove:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    
    name = name.strip()
    
    # Handle "County" suffix
    if not re.search(r"\bcounty\b", name, re.IGNORECASE):
        name = f"{name} County"
    
    # Title case but preserve special cases
    words = name.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in ("county", "of", "the"):
            result.append(lower.capitalize() if lower == "county" else lower)
        elif "-" in word:
            # Handle hyphenated names like Miami-Dade
            result.append("-".join(part.capitalize() for part in word.split("-")))
        elif word.startswith("st.") or word.lower() == "st":
            result.append("St.")
        else:
            result.append(word.capitalize())
    
    return " ".join(result)


def extract_county_from_path(path: str) -> str:
    """Extract and normalize county name from a file path."""
    import os
    
    # Get the parent directory name (county folder)
    parts = path.replace("\\", "/").split("/")
    
    # Find the county folder (usually parent of PDF file)
    for part in reversed(parts):
        if part.endswith(".pdf"):
            continue
        if part and not part.startswith("."):
            return normalize_county_name(part)
    
    return ""


def match_county_names(name1: str, name2: str) -> bool:
    """Check if two county names refer to the same county."""
    n1 = normalize_county_name(name1)
    n2 = normalize_county_name(name2)
    return n1.lower() == n2.lower()


def get_canonical_counties() -> list:
    """Return list of all 67 Florida county names in canonical form."""
    return [
        "Alachua County", "Baker County", "Bay County", "Bradford County",
        "Brevard County", "Broward County", "Calhoun County", "Charlotte County",
        "Citrus County", "Clay County", "Collier County", "Columbia County",
        "DeSoto County", "Dixie County", "Duval County", "Escambia County",
        "Flagler County", "Franklin County", "Gadsden County", "Gilchrist County",
        "Glades County", "Gulf County", "Hamilton County", "Hardee County",
        "Hendry County", "Hernando County", "Highlands County", "Hillsborough County",
        "Holmes County", "Indian River County", "Jackson County", "Jefferson County",
        "Lafayette County", "Lake County", "Lee County", "Leon County",
        "Levy County", "Liberty County", "Madison County", "Manatee County",
        "Marion County", "Martin County", "Miami-Dade County", "Monroe County",
        "Nassau County", "Okaloosa County", "Okeechobee County", "Orange County",
        "Osceola County", "Palm Beach County", "Pasco County", "Pinellas County",
        "Polk County", "Putnam County", "St. Johns County", "St. Lucie County",
        "Santa Rosa County", "Sarasota County", "Seminole County", "Sumter County",
        "Suwannee County", "Taylor County", "Union County", "Volusia County",
        "Wakulla County", "Walton County", "Washington County"
    ]


def find_matching_county(name: str, canonical_list: Optional[list] = None) -> Optional[str]:
    """
    Find the canonical county name that matches the given name.
    
    Returns None if no match found.
    """
    if canonical_list is None:
        canonical_list = get_canonical_counties()
    
    normalized = normalize_county_name(name)
    
    for canonical in canonical_list:
        if normalized.lower() == canonical.lower():
            return canonical
    
    # Try fuzzy matching for common variations
    name_lower = normalized.lower().replace(" county", "").strip()
    for canonical in canonical_list:
        canonical_lower = canonical.lower().replace(" county", "").strip()
        if name_lower == canonical_lower:
            return canonical
    
    return None
