#!/usr/bin/env python3
"""
Filter Aristotelian-style categorical syllogisms from a JSON dataset.

Input:
  A JSON array of objects like:
  {
      "id": "...",
      "syllogism": "...",
      "validity": true/false,
      "plausibility": true/false
  }

Output:
  A JSON file with extra fields:
    - form_flag: valid_form / invalid_form / uncertain
    - form_reason
    - parsed_clauses
    - normalized_terms
    - term_count

This script is conservative:
  - If it cannot confidently map a clause to A/E/I/O form, it returns "uncertain".
  - If it maps all three clauses but structural checks fail, it returns "invalid_form".

Usage:
  python filter_syllogism_form.py --input train_data.json --output train_data_flagged.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter


# =========================
# Data structures
# =========================

@dataclass
class ParsedClause:
    raw: str
    role: str  # "premise1", "premise2", "conclusion"
    form: Optional[str]  # A, E, I, O or None
    subject: Optional[str]
    predicate: Optional[str]
    confidence: str  # high / medium / low
    notes: List[str]


# =========================
# Text normalization
# =========================

DETERMINER_PREFIXES = [
    "a ", "an ", "the ", "any ", "some ", "every ", "each ",
    "all ", "no ", "not a single ", "every single ", "anything that is ",
    "something that is ", "everything that is ", "nothing that is ",
    "all of the things that are ", "every single thing that is ",
    "every single creature that is ", "every single person who is ",
    "anything that is ", "anything who is ", "anything which is ",
    "any person who is ", "each and every ", "all things that are ",
    "the group of ", "the set of ", "the category of ", "the class of ",
    "the entire group of ", "the entire category of ", "the items that are ",
    "the things that are ", "among the group of ", "among the people who are ", "that ",
    "thing ", "things, ", "not ", "It is a known fact that "
]

ROLE_HINTS = [
    "therefore", "thus", "hence", "consequently", "as a result",
    "it follows", "from this", "this means", "this implies",
    "we can conclude", "one must conclude", "the result of this is",
    "it is concluded", "it must be the case", "it logically follows",
    "therefore,", "thus,", "hence,", "consequently," "it is clear", "that", "it is a known fact that"
]

LEADING_FILLERS = [
    "it is true that ",
    "it is the case that ",
    "it is also the case that ",
    "it is a fact that ",
    "it is known that ",
    "it is completely false that ",
    "it is not true that ",
    "it is not the case that ",
    "it follows from this that ",
    "it is clear ",
    "it is clear that ",
    "this leads to the conclusion that ",
    "this necessarily suggests that ",
    "the only conclusion to draw is that ",
    "the only logical conclusion is that ",
    "from this, we can deduce that ",
    "from this, it follows that ",
    "it follows that ",
    "based on this, it must be the case that ",
    "a logical consequence of this is that ",
    "one can infer from this that ",
    "we must conclude that ",
    "therefore, ",
    "thus, ",
    "hence, ",
    "consequently, ",
    "as a result, ",
    "Consequently,",
    "consequently,",
    "this means that ",
    "hence",
    "hence,",
    "it can be concluded ",
    "one must conclude ",
    "this proves that ",
    "the conclusion is that ",
    "is inescapable",
    "as such, ",
    "it must be the case that ",
    "this demonstrates that ",
    "from this, it can be stated that ",
    "the only conclusion is that ",
    "it is therefore true that ",
    "it is a logical necessity that ",
    "it is concluded that ",
    "we can conclude that ",
    "from these facts, it is clear that ",
    "it's the case that ",
    "it is the case that ",
    "it is thus the case that ",
    "thus, ",
    "a conclusion that can be drawn from this is that ",
    "it can be deduced that some things that ",
    "it can be said that ",
    "it can be deduced that ",
    "the implication is that ",
    "this implies that ",
    "it is implied that ",
    "we can conclude that ",
    "it must be that ",
    "it is necessarily concluded that ",
    "from these facts, it is clear that ",
    "it's the case that ",
    "it is the case that ",
    "the conclusion that ",
    "it is necessarily true that ",
    "it is therefore the case that ",
    "it must follow that ",
    "it is suggested that ",
    "it logically follows that ",
    "it therefore stands that ",
    "it is necessarily the case that ",
    "this logically implies that ",
    "it is therefore a fact that ",
    "this necessitates that ",
    "it is therefore the case that ",
    "it has been asserted that ",
    "it is certain that ",
    "it necessarily follows that ",
    "the consequence is that ",
    "it can be concluded ",
    "it must be true that ",
    "it has been said that ",
    "this necessarily means that ",
    "we can conclude from this that ",
    "it can be logically concluded that ",
    "it is argued that ",
    "we can derive that ",
    "this has led to the conclusion that ",
    "based on this, ",
    "this suggests that ",
    "it can be claimed that ",
    "this allows us to state that ",
    "from these statements, it can be concluded that ",
    "it is therefore a sound deduction that ",
    "one can therefore conclude that ",
    "we can logically conclude that ",
    "based on this, ",
    "this logically means that ",
    "we can logically derive that ",
    "we can validly conclude that ",
    "it is a logical consequence that ",
    "it is concluded that ",
    "it is possible to infer that ",
    "it must be true that ",
    "from these statements, ",
    "one can conclude that ",
    "from these statements, one can conclude that ",
    "the result of this is that ",
    "we can infer that ",
    "it is certain that ",
    "this necessarily implies that ",
    "it is a logical consequence that ",
    "this has led to the conclusion that ",
    "it is possible to conclude that ",
    "this has led to the conclusion that ",
    "it must be concluded that ",
    "one might conclude that ",
    "one may conclude that ",
    "one can conclude that ",
    "one could conclude that ",
    "one can therefore conclude that ",
    "it must therefore be true that ",
    "the consequence is that ",
    "we can therefore deduce that ",
    "we must accept that ",
    "we can deduce that ",
    "it is therefore suggested that ",
    "it has been proposed that ",
    "it can be concluded that ",
    "on this basis, it has been stated that ",
    "a valid inference is that ",
    "this logically entails that ",
    "a consequence is that ",
    "from these two facts, one can conclude that ",
    "based on this information, it is logically necessary that ",
    "it therefore follows that ",
    "this logically entails that ",
    "it can be concluded that ",
    "as a consequence, ",
    "we know for certain that ",
    "this logically suggests that ",
    "it is correct to infer that ",
    "for this reason, ",
    "a valid inference is that ",
    "it can be derived that ",
    "we know that ",
    "this necessitates the conclusion that ",
    "it must logically follow that ",
    "from this, it is concluded that ",
    "from this, it is suggested that ",
    "from this, we can logically conclude that ",
    "from this, we must conclude ",
    "directly that ",
    "logically that ",
    "it follows, then, that "
]


def clean_text(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_outer_punctuation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[,;:\-\s]+", "", text)
    text = re.sub(r"[,;:\-\s]+$", "", text)
    return text.strip()


def strip_leading_fillers(text: str) -> str:
    t = text.lower().strip()
    changed = True
    while changed:
        changed = False
        for prefix in sorted(LEADING_FILLERS, key=len, reverse=True):
            if t.startswith(prefix):
                t = t[len(prefix):].strip()
                changed = True
    return strip_outer_punctuation(t)


def normalize_term(term: str) -> str:
    """
    Conservative term normalization.
    We do NOT do aggressive semantic conflation.
    """
    t = term.lower().strip()
    t = re.sub(r"[\"'`]", "", t)
    t = re.sub(r"\s+", " ", t)
    t = strip_outer_punctuation(t)

    # Remove common leading determiners / wrappers repeatedly
    changed = True
    while changed:
        changed = False
        for pref in sorted(DETERMINER_PREFIXES, key=len, reverse=True):
            if t.startswith(pref):
                t = t[len(pref):].strip()
                changed = True

    # Light singularization for final noun token only
    # Avoid overdoing it.
    tokens = t.split()
    if tokens:
        last = tokens[-1]
        if len(tokens)>=2 and last == 'that':
            tokens = tokens[:-1]
            last = tokens[-1]
            
        if last in ["tomatoes", "potatoes", "buses", "muses", "fusses", "blueses"]:
            last = last[:-2]
        elif len(last) > 4 and last.endswith("ies"):
            last = last[:-3] + "y"
        elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
            last = last[:-1]
        
        tokens[-1] = last
    t = " ".join(tokens)

    # Normalize a few phrasings
    replacements = {
        "not a ": "",
        "not an ": "",
        "type of ": "",
        "kind of ": "",
        "class of ": "",
        "group of ": "",
        "set of ": "",
        "object that is ": "",
        "thing that is ": "",
        "creature that is ": "",
        "person who is ": "",
        "item which is ": "",
        "object which is ": "",
    }
    for a, b in replacements.items():
        if t.startswith(a):
            t = t[len(a):]

    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_clause_text(text: str) -> str:
    t = text.lower().strip()

    # remove discourse parentheticals
    t = re.sub(r",\s*without exception,\s*", " ", t)
    t = re.sub(r"\bwithout exception\b", " ", t)

    # normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def looks_like_relational_or_noncategorical(term: str) -> bool:
    """
    Reject terms that suggest the clause is not a simple unary category.
    Very conservative.
    """
    bad_markers = [
        " and ", " or ", " if ", " then ", " because ",
        " who ", " which ", " that ",
    ]
    # Allow some embedded "that are ..." patterns before normalization, but after
    # normalization terms should be simple-ish noun phrases.
    for m in bad_markers:
        if m in term:
            return True
    return False


# =========================
# Clause splitting
# =========================

def split_into_clauses(syllogism: str) -> List[str]:
    text = clean_text(syllogism)

    # First split by sentence punctuation.
    parts = re.split(r"[.!?]+", text)
    parts = [strip_outer_punctuation(p) for p in parts if p.strip()]

    # Sometimes the conclusion marker appears inside a longer sentence.
    # If we don't get 3 parts, try splitting on conclusion markers.
    if len(parts) != 3:
        lowered = text.lower()
        marker_positions = []
        for marker in ROLE_HINTS:
            m = lowered.find(marker)
            if m > 0:
                marker_positions.append((m, marker))
        if marker_positions:
            marker_positions.sort()
            m, marker = marker_positions[0]
            before = text[:m].strip(" ,;")
            after = text[m:].strip(" ,;")
            premise_parts = re.split(r"[.!?;]+", before)
            premise_parts = [strip_outer_punctuation(p) for p in premise_parts if p.strip()]
            if len(premise_parts) >= 2:
                premises = premise_parts[:2]
                conclusion = strip_outer_punctuation(after)
                return premises + [conclusion]

    return parts


# =========================
# Clause parsing
# =========================

def parse_clause(raw_clause: str, role: str) -> ParsedClause:
    raw_clause = clean_text(raw_clause)
    clause=raw_clause
    ignore=''
    
    #Transform goofy clauses of type "All xxx that are [subject] are [predicate]"
    #into easy to handle clauses of type "All [subject] are [predicate]"
    
    if ", without exception, " in raw_clause:
        clause = re.sub(r",\s*without\s+exception,\s*", " ", raw_clause, flags=re.I)
        #print(clause)
        
    if ", surprisingly, " in raw_clause:
        clause = re.sub(r",\s*surprisingly,\s*", " ", raw_clause, flags=re.I)
        #print(clause)
    
    if "that is " in raw_clause and "There " not in raw_clause and "there " not in raw_clause:
        marker = "that is "
        i = raw_clause.find(marker)

        if i != -1:
            start = raw_clause.rfind(" ", 0, i - 1) + 1
            clause = raw_clause[:start] + raw_clause[i + len(marker):]
            ignore = strip_leading_fillers(raw_clause[:start])
            ignore = normalize_clause_text(raw_clause[:start])
    elif "that are " in raw_clause and "There " not in raw_clause and "there " not in raw_clause:
        marker = "that are "
        i = raw_clause.find(marker)

        if i != -1:
            start = raw_clause.rfind(" ", 0, i - 1) + 1
            clause = raw_clause[:start] + raw_clause[i + len(marker):]
            ignore = strip_leading_fillers(raw_clause[:start])
            ignore = normalize_clause_text(raw_clause[:start])
    elif "who is " in raw_clause and "There " not in raw_clause and "there " not in raw_clause:
        marker = "who is "
        i = raw_clause.find(marker)

        if i != -1:
            start = raw_clause.rfind(" ", 0, i - 1) + 1
            clause = raw_clause[:start] + raw_clause[i + len(marker):]
            ignore = strip_leading_fillers(raw_clause[:start])
            ignore = normalize_clause_text(raw_clause[:start])
    elif "who are " in raw_clause and "There " not in raw_clause and "there " not in raw_clause:
        marker = "who are "
        i = raw_clause.find(marker)

        if i != -1:
            start = raw_clause.rfind(" ", 0, i - 1) + 1
            clause = raw_clause[:start] + raw_clause[i + len(marker):]
            ignore = strip_leading_fillers(raw_clause[:start])
            ignore = normalize_clause_text(raw_clause[:start])
    elif "which is " in raw_clause and "There " not in raw_clause and "there " not in raw_clause:
        marker = "which is "
        i = raw_clause.find(marker)

        if i != -1:
            start = raw_clause.rfind(" ", 0, i - 1) + 1
            clause = raw_clause[:start] + raw_clause[i + len(marker):]
            ignore = strip_leading_fillers(raw_clause[:start])
            ignore = normalize_clause_text(raw_clause[:start])
    elif "which are " in raw_clause and "There " not in raw_clause and "there " not in raw_clause:
        marker = "which are "
        i = raw_clause.find(marker)

        if i != -1:
            start = raw_clause.rfind(" ", 0, i - 1) + 1
            clause = raw_clause[:start] + raw_clause[i + len(marker):]
            ignore = strip_leading_fillers(raw_clause[:start])
            ignore = normalize_clause_text(raw_clause[:start])
    
    if ignore != '':
        ignore=ignore.split()[-1]

        
    clause = strip_leading_fillers(clause)
    clause = normalize_clause_text(clause)
    notes: List[str] = []

    # Remove some discourse wrappers
    clause = re.sub(r"^(therefore|thus|hence|consequently|as a result|it follows|from this|this means)\b[:, ]*", "", clause)
    clause = strip_outer_punctuation(clause)

    # Reject obvious non-categorical conditionals/relations
    if " if " in clause or " either " in clause or " unless " in clause:
        return ParsedClause(raw_clause, role, None, None, None, "low", ["conditional/disjunctive phrasing"])

    # Pattern order matters.

    patterns: List[Tuple[str, str, re.Pattern]] = []
    
    # E: No S are P
    patterns.append((
        "E",
        "universal negative",
        re.compile(
            rf"^(?:it is necessarily true that no|it is also true that no|it is true that no|it is true that none|there is not a single|it is impossible for|there is no|there are no|it is true that no|nothing that is|not a single|not one single|there are no|there exist no|it is impossible for|none of|no instance of|absolutely nothing|absolutely no|are not considered|is not considered|considered|absolutely|no one|not one|none|no)\s+(?:(?:{ignore})\s+)?+(.+?)\s+(?:that are|that is|which are|which is|who are|who is|can be considered|can be called|can be classified as|can be|to be|are|is)\s+(.+)$",
            re.I
        )
    ))
    

    # A: All S are P

    
    patterns.append((
        "A",
        "universal affirmative",
        re.compile(
            rf"^(?:It is a known fact that|anything and everything|anything and|it is necessarily true that|it is true that|anything that can be called|everything that is|every single|anything that is|everything that is|all things that are|all of the things that are|all items of|all instances of|any and all|all of the considered|are considered|is considered|all of|considered|any|all|every|each)\s+(?:(?:{ignore})\s+)?+(.+?)\s+(?:are also|is is also|a is also|is also|can be considered|is considered|are considered|can be called|belong to|belong|are|is(?:s)? to the class of|can be classified|is also|is a type of|is a kind of|is an?)\s+(.+)$",
            re.I
        )
    ))


    
    # O: Some S are not P
    patterns.append((
        "O",
        "particular negative",
        re.compile(
            rf"^(?:there is a subset of|a number of|a portion of|there are some|there exist some|there exists at least one|at least one|a certain quantity of|certain quantity of|certain number of|a certain number of|certain|some|a few)\s+(?:(?:{ignore})\s+)?+(.+?)\s+(?:are|is)\s+not\s+(.+)$",
            re.I
        )
    ))

    # I: Some S are P
    patterns.append((
        "I",
        "particular affirmative",
        re.compile(
            rf"^(?:there is a subset of|it is a known fact that some|there exists a group of|a subset of|a few of the things known as|a select few|a certain quantity of|certain quantity of|a certain number of|a number of|a portion of|there are some|there exist some|there exists at least one|at least one|certain|certain number of|a certain number of|a subset of|there exist|there are a few|there is a few|there is|there are|a few|among the|among|some)\s+(?:(?:{ignore})\s+)?+(.+?)\s+(?:are classified as|that is also|that are also|is composed of|are composed of|are also|is also|that is|that are|who is|who are|which is|which are|is considered|are considered|can be considered|can be called|is included in|is in|are|is)\s+(.+)$",
            re.I
        )
    ))

    

    # Common transformed patterns

    # "S are in no way P" -> E
    patterns.append((
        "E",
        "universal negative transformed",
        re.compile(r"^(.+?)\s+are\s+in\s+no\s+way\s+(.+)$", re.I)
    ))

    # "S is never P" / "S are never P" -> E
    patterns.append((
        "E",
        "universal negative transformed",
        re.compile(r"^(.+?)\s+(?:is|are)\s+never\s+(.+)$", re.I)
    ))

    # "S cannot be classified as P" -> E  (treat as No S are P)
    patterns.append((
        "E",
        "universal negative transformed",
        re.compile(r"^(.+?)\s+cannot\s+be\s+classified\s+as\s+(.+)$", re.I)
    ))

    # "Every S is not P" -> ambiguous in English, but in this dataset it often means No S are P.
    patterns.append((
        "E",
        "ambiguous every-not interpreted as no",
        re.compile(r"^(?:every single|every|all)\s+(.+?)\s+(?:is|are)\s+not\s+(.+)$", re.I)
    ))

    # "Not all S are P" -> O
    patterns.append((
        "O",
        "particular negative transformed",
        re.compile(r"^not\s+all\s+(.+?)\s+(?:are|is)\s+(.+)$", re.I)
    ))
    
    patterns.append((
        "O",
        "particular negative transformed",
        re.compile(r"^a protion of|a certain amount of\s+(.+?)\s+(?:are|is)\s+not\s+(.+)$", re.I)
    ))

    # "It is not true that any S is P" -> E
    patterns.append((
        "E",
        "universal negative transformed",
        re.compile(r"^(?:it is not true that|it is not the case that|it is completely false that)\s+any\s+(.+?)\s+(?:is|are)\s+(.+)$", re.I)
    ))

    # "There is no S that is P" -> E
    patterns.append((
        "E",
        "universal negative transformed",
        re.compile(r"^there\s+is\s+no\s+(.+?)\s+that\s+(?:is|are)\s+(.+)$", re.I)
    ))

    # "No S can be P" -> E
    patterns.append((
        "E",
        "universal negative transformed",
        re.compile(r"^no\s+(.+?)\s+can\s+be\s+(.+)$", re.I)
    ))

    # "Some S fail to be P" -> O
    patterns.append((
        "O",
        "particular negative transformed",
        re.compile(r"^(?:some|there are some|there exist some)\s+(.+?)\s+fail\s+to\s+be\s+(.+)$", re.I)
    ))

    # "There are S, and some of them are P" -> I
    patterns.append((
        "I",
        "existential transformed",
        re.compile(r"^there\s+are\s+(.+?),\s+and\s+some\s+of\s+them\s+are\s+(.+)$", re.I)
    ))

    # "There are S, and some of them are not P" -> O
    patterns.append((
        "O",
        "existential transformed",
        re.compile(r"^there\s+are\s+(.+?),\s+and\s+some\s+of\s+them\s+are\s+not\s+(.+)$", re.I)
    ))
    
    # E: S are/is not P
    patterns.append((
        "E",
        "universal negative are not",
        re.compile(
            r"^(.+?)\s+(?:are|is)\s+(?:not|in no way|inacceptable)\s+(.+)$",
            re.I
        )
    ))
    
    # E: S are/is P
    patterns.append((
        "A",
        "universal affirmative are",
        re.compile(
            r"^(.+?)\s+(?:are also|is also|are|is)\s+(.+)$",
            re.I
        )
    ))
    

    


    for form, description, pattern in patterns:
        m = pattern.match(clause)
        if not m:
            continue

        left = normalize_term(m.group(1))
        right = normalize_term(m.group(2))

        # For "It is impossible for S to be P", captured group(1) may still be odd.
        left = re.sub(r"^a\s+", "", left)
        right = re.sub(r"^a\s+", "", right)
        

        
        
        
        #I don't want 'thing' to be subject nor predicate unless necessary
        if True:
            if ('thing' in left or 'things' in left) and len(left.split())>1:
                tokens = left.split()
                left = ''
                for t in tokens[1:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('thing' in right or 'things' in right) and len(right.split())>1:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('piece of ' in left or "pieces of " in left) and len(left.split())>2:
                tokens = left.split()
                left = ''
                for t in tokens[2:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ("piece of " in right or "pieces of " in right) and len(right.split())>2:
                tokens = right.split()
                right = ''
                for t in tokens[2:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ("kind of " in left or "kinds of " in left) and len(left.split())>2:
                tokens = left.split()
                left = ''
                for t in tokens[2:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ("kind of " in right or "kinds of " in right) and len(right.split())>2:
                tokens = right.split()
                right = ''
                for t in tokens[2:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ("type of " in left or "types of " in left) and len(left.split())>2:
                tokens = left.split()
                left = ''
                for t in tokens[2:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ("type of " in right or "types of " in right) and len(right.split())>2:
                tokens = right.split()
                right = ''
                for t in tokens[2:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ("sort of " in left or "sorts of " in left) and len(left.split())>2:
                tokens = left.split()
                left = ''
                for t in tokens[2:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ("sort of " in right or "sorts of " in right) and len(right.split())>2:
                tokens = right.split()
                right = ''
                for t in tokens[2:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('object that is' in left or 'objects that are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('object that is' in right or 'objects that are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('object which is' in left or 'objects which are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('object which is' in right or 'objects wchich are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('object that is' in left or 'objects that are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('thing that is' in right or 'things that are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('thing that is' in left or 'things that are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('thing that is' in right or 'things that are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('thing which is' in left or 'things which are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('thing which is' in right or 'things which are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('creature that is' in left or 'creatures that are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('creature that is' in right or 'creatures that are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('creature which is' in left or 'creatures which are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('creature which is' in right or 'creatures which are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('item that is' in left or 'items that are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('item that is' in right or 'items that are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('item which is' in left or 'items which are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('item which is' in right or 'people which are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('person that is' in left or 'people that are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('person that is' in right or 'people that are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('person which is' in left or 'people which are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('person which is' in right or 'people which are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('person who is' in left or 'people who are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('person who is' in right or 'people who are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('individual that is' in left or 'individuals that are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('individual that is' in right or 'individuals that are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
                
            if ('individual which is' in left or 'individuals which are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('individual which is' in right or 'individuals which are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('individual who is' in left or 'individuals who are' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('individual who is' in right or 'individuals who are' in right) and len(right.split())>3:
                tokens = right.split()
                right = ''
                for t in tokens[1:]:
                    right = right + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('one of the' in left) and len(left.split())>3:
                tokens = left.split()
                left = ''
                for t in tokens[3:]:
                    left = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('one of the' in right) and len(left.split())>3:
                tokens = left.split()
                right = ''
                for t in tokens[3:]:
                    right = left + t + " "
                left = normalize_term(left)
                right = normalize_term(right)
            
            if ('also' in left) and len(left.split())>1:
                marker = "also"
                i = left.find(marker)
                print(left)

                if i != -1:
                    start = left.rfind(" ", 0, i) + 1
                    left = left[:start] + left[i + len(marker):]
                left = normalize_term(left)
                print(left)
                right = normalize_term(right)
            
            if ('also' in right) and len(right.split())>1:
                marker = "also"
                i = right.find(marker)
                print(right)

                if i != -1:
                    start = right.rfind(" ", 0, i) + 1
                    right = right[:start] + right[i + len(marker):]
                left = normalize_term(left)
                right = normalize_term(right)
                print(right)
            
            if ('single' in left) and len(left.split())>1:
                marker = "single"
                i = left.find(marker)
                print(left)

                if i != -1:
                    start = left.rfind(" ", 0, i) + 1
                    left = left[:start] + left[i + len(marker):]
                left = normalize_term(left)
                print(left)
                right = normalize_term(right)
            
            if ('single' in right) and len(right.split())>1:
                marker = "single"
                i = right.find(marker)
                print(right)

                if i != -1:
                    start = right.rfind(" ", 0, i) + 1
                    right = right[:start] + right[i + len(marker):]
                left = normalize_term(left)
                right = normalize_term(right)
                print(right)


        
        
            if len(left.split())>1 and right is None:
                splits = left.split()[0]
                left, right = splits[0], splits[1]
                left = normalize_term(left)
                right = normalize_term(right)
            
            if len(right.split())>1 and left is None:
                splits = right.split()[0]
                left, right = splits[0], splits[1]
                left = normalize_term(left)
                right = normalize_term(right)
            
            if len(left.split())>1:
                left = left.split()[0]
                left = normalize_term(left)
                right = normalize_term(right)
            
            if len(right.split())>1:
                right = right.split()[0]
                left = normalize_term(left)
                right = normalize_term(right)
        

        if not left or not right:
            return ParsedClause(raw_clause, role, None, None, None, "low", [f"{description}: empty term after normalization"])

        if left == right:
            notes.append("subject and predicate normalize to same term")

        # Conservative complexity filter
        if looks_like_relational_or_noncategorical(left) or looks_like_relational_or_noncategorical(right):
            return ParsedClause(raw_clause, role, None, None, None, "low", [f"{description}: term looks non-categorical"])

        confidence = "high"
        if "ambiguous" in description or "transformed" in description:
            confidence = "medium"
            
        #if role == 'conclusion':
            #print(clause)

        return ParsedClause(
            raw=raw_clause,
            role=role,
            form=form,
            subject=left,
            predicate=right,
            confidence=confidence,
            notes=notes + [description]
        )
    
    #if role == 'conclusion':
        #print(clause)
    return ParsedClause(raw_clause, role, None, None, None, "low", ["no categorical pattern matched"])


# =========================
# Syllogism-level checks
# =========================

def count_terms(parsed: List[ParsedClause]) -> Counter:
    c = Counter()
    for p in parsed:
        if p.subject:
            c[p.subject] += 1
        if p.predicate:
            c[p.predicate] += 1
    return c


def conclusion_marker_present(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ROLE_HINTS)


def check_syllogistic_form(parsed: List[ParsedClause], original_text: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Return (flag, reason, debug_info)
    """

    debug: Dict[str, Any] = {}

    if len(parsed) != 3:
        return "invalid_form", f"expected 3 clauses, found {len(parsed)}", debug

    # All three must parse
    if any(p.form is None for p in parsed):
        # If at least one clause is clearly unparsable, prefer uncertain.
        failed_roles = [p.role for p in parsed if p.form is None]
        return "uncertain", f"could not confidently parse clause(s): {failed_roles}", debug

    # Label roles
    premise1, premise2, conclusion = parsed

    # Soft check for conclusion marker
    if not conclusion_marker_present(original_text):
        debug["warning"] = "no explicit conclusion marker detected"

    # Exactly 3 distinct terms for strict categorical syllogism
    terms = count_terms(parsed)
    debug["term_frequencies"] = dict(terms)
    distinct_terms = list(terms.keys())
    if len(distinct_terms) < 3:
        return "uncertain", f"fewer than 3 distinct terms after normalization ({len(distinct_terms)})", debug
    if len(distinct_terms) > 3:
        # Could be bad normalization or a genuine non-syllogism.
        return "uncertain", f"more than 3 distinct terms after normalization ({len(distinct_terms)})", debug

    # Conclusion terms must be exactly two distinct terms
    if conclusion.subject is None or conclusion.predicate is None:
        return "uncertain", "conclusion did not parse cleanly", debug
    if conclusion.subject == conclusion.predicate:
        return "invalid_form", "conclusion subject and predicate collapse to same term", debug

    conclusion_terms = {conclusion.subject, conclusion.predicate}
    all_terms = set(distinct_terms)
    middle_terms = list(all_terms - conclusion_terms)

    if len(middle_terms) != 1:
        return "invalid_form", "could not identify unique middle term", debug

    middle = middle_terms[0]
    debug["middle_term"] = middle

    # Middle term must appear in both premises and not in conclusion
    p1_terms = {premise1.subject, premise1.predicate}
    p2_terms = {premise2.subject, premise2.predicate}

    if middle not in p1_terms or middle not in p2_terms:
        return "invalid_form", "middle term does not appear in both premises", debug

    if middle in conclusion_terms:
        return "invalid_form", "middle term appears in conclusion", debug

    # Each premise should contain the middle term plus one conclusion term
    other_terms_p1 = p1_terms - {middle}
    other_terms_p2 = p2_terms - {middle}
    if len(other_terms_p1) != 1 or len(other_terms_p2) != 1:
        return "invalid_form", "a premise does not contain exactly two terms", debug

    if (other_terms_p1 | other_terms_p2) != conclusion_terms:
        return "invalid_form", "premises do not connect the two conclusion terms through one middle term", debug

    # Reject too many low-confidence transforms
    lowish = sum(1 for p in parsed if p.confidence == "low")
    mediumish = sum(1 for p in parsed if p.confidence == "medium")
    debug["confidence_counts"] = {
        "low": lowish,
        "medium": mediumish,
        "high": sum(1 for p in parsed if p.confidence == "high")
    }

    if lowish > 0:
        return "uncertain", "at least one clause had low-confidence parsing", debug
    if mediumish >= 2:
        return "uncertain", "too many clauses required transformed/ambiguous parsing", debug

    return "valid_form", "matches strict 3-term categorical syllogism structure", debug


# =========================
# Dataset processing
# =========================

def analyze_record(record: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(record)

    syllogism = record.get("syllogism", "")
    if not isinstance(syllogism, str) or not syllogism.strip():
        result["form_flag"] = "invalid_form"
        result["form_reason"] = "missing or empty syllogism text"
        result["parsed_clauses"] = []
        result["normalized_terms"] = []
        result["term_count"] = 0
        return result

    clauses = split_into_clauses(syllogism)

    roles = ["premise1", "premise2", "conclusion"]
    parsed: List[ParsedClause] = []
    for i, clause in enumerate(clauses[:3]):
        parsed.append(parse_clause(clause, roles[i]))

    # If clause count is off, still record what we saw
    if len(clauses) != 3:
        result["form_flag"] = "invalid_form"
        result["form_reason"] = f"expected 3 clauses, found {len(clauses)}"
        result["parsed_clauses"] = [asdict(p) for p in parsed]
        term_counter = count_terms(parsed)
        result["normalized_terms"] = sorted(term_counter.keys())
        result["term_count"] = len(term_counter)
        return result

    flag, reason, debug = check_syllogistic_form(parsed, syllogism)

    result["form_flag"] = flag
    result["form_reason"] = reason
    result["parsed_clauses"] = [asdict(p) for p in parsed]

    term_counter = count_terms(parsed)
    result["normalized_terms"] = sorted(term_counter.keys())
    result["term_count"] = len(term_counter)
    result["form_debug"] = debug

    return result


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(r.get("form_flag", "unknown") for r in records)
    return {
        "total": len(records),
        "counts": dict(counts)
    }


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='C:/Users/stepi/DRL_assignments/LLMs/train_data_s2.json', help="Path to input JSON file")
    parser.add_argument("--output", default = 'C:/Users/stepi/DRL_assignments/LLMs/output_s2.json', help="Path to output JSON file")
    parser.add_argument("--summary", default='C:/Users/stepi/DRL_assignments/LLMs/summary_s2.json', help="Optional path to summary JSON")
    args = parser.parse_args()

    data = load_json(args.input)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    flagged = [analyze_record(rec) for rec in data]
    save_json(args.output, flagged)

    summary = summarize(flagged)
    if args.summary:
        save_json(args.summary, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()