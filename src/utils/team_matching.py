"""Team name matching utilities for LoL esports.

Provides fuzzy matching between different naming conventions
(Polymarket, Oracle's Elixir, Liquipedia, etc.)
"""

import re
from functools import lru_cache

# Common team name aliases and abbreviations
TEAM_ALIASES: dict[str, str] = {
    # Korea (LCK)
    'dwg kia': 'dplus kia',
    'damwon': 'dplus kia',
    'damwon gaming': 'dplus kia',
    'dk': 'dplus kia',
    'skt': 't1',
    'skt t1': 't1',
    'sk telecom': 't1',
    'sk telecom t1': 't1',
    'hle': 'hanwha life esports',
    'hanwha life': 'hanwha life esports',
    'drx': 'drx',
    'kt': 'kt rolster',
    'kdf': 'kwangdong freecs',
    'freecs': 'kwangdong freecs',
    'ns': 'nongshim red force',
    'nongshim': 'nongshim red force',
    'gen.g': 'gen.g',
    'geng': 'gen.g',
    'sandbox': 'liiv sandbox',
    'lsb': 'liiv sandbox',
    'brion': 'oksavingsbank brion',
    'fearx': 'bnk fearx',
    'bnk': 'bnk fearx',

    # China (LPL)
    'edg': 'edward gaming',
    'edward': 'edward gaming',
    'rng': 'royal never give up',
    'royal': 'royal never give up',
    'fpx': 'funplus phoenix',
    'funplus': 'funplus phoenix',
    'ig': 'invictus gaming',
    'invictus': 'invictus gaming',
    'tes': 'top esports',
    'jdg': 'jd gaming',
    'jingdong': 'jd gaming',
    'lng': 'lng esports',
    'blg': 'bilibili gaming',
    'bilibili': 'bilibili gaming',
    'wbg': 'weibo gaming',
    'weibo': 'weibo gaming',
    'omg': 'oh my god',
    'we': 'team we',
    'tt': 'thundertalk gaming',
    'thundertalk': 'thundertalk gaming',
    'al': "anyone's legend",
    'anyone': "anyone's legend",
    'up': 'ultra prime',
    'lgd': 'lgd gaming',
    'ra': 'rare atom',
    'v5': 'victory five',

    # North America (LCS)
    'c9': 'cloud9',
    'tl': 'team liquid',
    'liquid': 'team liquid',
    '100t': '100 thieves',
    '100 thieves': '100 thieves',
    'eg': 'evil geniuses',
    'flyquest': 'flyquest',
    'fq': 'flyquest',
    'dig': 'dignitas',
    'gg': 'golden guardians',
    'imt': 'immortals',
    'clg': 'counter logic gaming',
    'tsm': 'tsm',

    # Europe (LEC)
    'kc': 'karmine corp',
    'karmine': 'karmine corp',
    'fnc': 'fnatic',
    'g2': 'g2 esports',
    'mad': 'mad lions',
    'vit': 'team vitality',
    'vitality': 'team vitality',
    'bds': 'team bds',
    'sk': 'sk gaming',
    'ast': 'astralis',
    'xl': 'excel esports',
    'excel': 'excel esports',
    'msf': 'misfits gaming',
    'msfx': 'misfits gaming',
    'misfits': 'misfits gaming',
    'rge': 'rogue',
    'heretics': 'team heretics',
    'giantx': 'giantx',
    'gx': 'giantx',
    'koi': 'movistar koi',
    'movistar': 'movistar koi',

    # Brazil (CBLOL)
    'loud': 'loud',
    'pain': 'pain gaming',
    'red': 'red canids',
    'red canids': 'red canids',
    'keyd': 'vivo keyd stars',
    'keyd stars': 'vivo keyd stars',
    'fluxo': 'fluxo w7m',
    'los grandes': 'los grandes',

    # Other regions
    'psg': 'psg talon',
    'talon': 'psg talon',
    'cfx': 'cfx gaming',
    'gam': 'gam esports',
    'skt academy': 't1 esports academy',
    't1 academy': 't1 esports academy',

    # CIS/EMEA (new LEC teams)
    'navi': 'natus vincere',
    'natus vincere': 'natus vincere',
    "na'vi": 'natus vincere',
    'shifters': 'shifters',
    'ratones': 'los ratones',
    'los ratones': 'los ratones',
}

# Words to strip from team names for matching
STRIP_WORDS = [
    'esports', 'esport', 'gaming', 'team', 'org', 'club',
    'academy', 'challengers', 'youth', 'rising', 'rookies',
]


def normalize_team_name(name: str) -> str:
    """Normalize a team name for matching.

    Args:
        name: Raw team name from any source

    Returns:
        Normalized lowercase team name
    """
    if not name:
        return ""

    name_lower = name.lower().strip()

    # Remove common suffixes like (BO3), (BO5), etc.
    name_lower = re.sub(r'\s*\(bo\d+\)\s*$', '', name_lower)

    # Check direct alias match first
    if name_lower in TEAM_ALIASES:
        return TEAM_ALIASES[name_lower]

    # Check if any alias is contained in the name
    for alias, canonical in TEAM_ALIASES.items():
        if alias in name_lower:
            return canonical

    return name_lower


def get_team_variants(team_name: str) -> list[str]:
    """Get all possible variants of a team name for fuzzy matching.

    Args:
        team_name: Normalized team name

    Returns:
        List of possible name variants to search for
    """
    variants = [team_name]

    # Add version without common suffixes
    for word in STRIP_WORDS:
        stripped = team_name.replace(f' {word}', '').strip()
        if stripped and stripped != team_name:
            variants.append(stripped)

    # Add version with just the first word (often the main identifier)
    first_word = team_name.split()[0] if team_name else ""
    if first_word and len(first_word) > 2 and first_word not in variants:
        variants.append(first_word)

    return list(set(variants))


@lru_cache(maxsize=1000)
def match_team_name(query: str, candidates: tuple[str, ...]) -> str | None:
    """Find the best matching team name from a list of candidates.

    Args:
        query: Team name to search for
        candidates: Tuple of possible team names (must be tuple for caching)

    Returns:
        Best matching candidate or None if no match found
    """
    if not query or not candidates:
        return None

    query_norm = normalize_team_name(query)
    query_variants = get_team_variants(query_norm)

    # First pass: exact match on normalized names
    for candidate in candidates:
        cand_norm = normalize_team_name(candidate)
        if cand_norm == query_norm:
            return candidate

    # Second pass: check if any variant matches
    for candidate in candidates:
        cand_norm = normalize_team_name(candidate)
        cand_variants = get_team_variants(cand_norm)

        for qv in query_variants:
            if qv in cand_variants:
                return candidate

    # Third pass: substring matching
    for candidate in candidates:
        cand_norm = normalize_team_name(candidate)

        # Check if query is contained in candidate or vice versa
        if query_norm in cand_norm or cand_norm in query_norm:
            return candidate

        # Check variants
        for qv in query_variants:
            if qv in cand_norm or cand_norm in qv:
                return candidate

    return None


def find_team_in_dataframe(df, team_name: str, team1_col: str = 'team1', team2_col: str = 'team2'):
    """Find rows in a dataframe where a team appears.

    Args:
        df: Pandas DataFrame with team columns
        team_name: Team name to search for
        team1_col: Name of first team column
        team2_col: Name of second team column

    Returns:
        Tuple of (as_team1_df, as_team2_df, is_team1_most_recent)
    """
    import pandas as pd

    team_norm = normalize_team_name(team_name)
    team_variants = get_team_variants(team_norm)

    # Try exact match first
    as_team1 = df[df[team1_col].apply(normalize_team_name) == team_norm]
    as_team2 = df[df[team2_col].apply(normalize_team_name) == team_norm]

    # Try variant matching if no exact match
    if len(as_team1) == 0 and len(as_team2) == 0:
        for variant in team_variants:
            as_team1 = df[df[team1_col].apply(normalize_team_name).str.contains(variant, na=False, regex=False)]
            as_team2 = df[df[team2_col].apply(normalize_team_name).str.contains(variant, na=False, regex=False)]
            if len(as_team1) > 0 or len(as_team2) > 0:
                break

    # Try reverse containment
    if len(as_team1) == 0 and len(as_team2) == 0:
        mask1 = df[team1_col].apply(lambda x: normalize_team_name(x) in team_norm if pd.notna(x) else False)
        mask2 = df[team2_col].apply(lambda x: normalize_team_name(x) in team_norm if pd.notna(x) else False)
        as_team1 = df[mask1]
        as_team2 = df[mask2]

    # Determine which is most recent
    is_team1_most_recent = True
    if len(as_team1) > 0 and len(as_team2) > 0:
        if 'date' in df.columns:
            t1_date = as_team1['date'].max()
            t2_date = as_team2['date'].max()
            is_team1_most_recent = t1_date >= t2_date

    return as_team1, as_team2, is_team1_most_recent
