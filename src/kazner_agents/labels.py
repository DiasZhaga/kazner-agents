"""The KazNERD label set: 25 entity types, used as IOB2 labels (B-TYPE, I-TYPE, O).

The list was read from ../ner-project/data (identical in train, valid and test); a test checks
it against the data when the folder is available. The short descriptions go into the LLM prompt.
"""

from typing import Literal, get_args

EntityType = Literal[
    "ADAGE",
    "ART",
    "CARDINAL",
    "CONTACT",
    "DATE",
    "DISEASE",
    "EVENT",
    "FACILITY",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOCATION",
    "MISCELLANEOUS",
    "MONEY",
    "NON_HUMAN",
    "NORP",
    "ORDINAL",
    "ORGANISATION",
    "PERCENTAGE",
    "PERSON",
    "POSITION",
    "PRODUCT",
    "PROJECT",
    "QUANTITY",
    "TIME",
]

ENTITY_TYPES: tuple[str, ...] = get_args(EntityType)

# Every valid IOB2 label: "O" plus B-/I- for each type (51 labels).
IOB2_LABELS: frozenset[str] = frozenset(
    ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
)

DESCRIPTIONS: dict[str, str] = {
    "ADAGE": "proverbs and sayings",
    "ART": "titles of works: books, songs, films, TV programmes, newspapers",
    "CARDINAL": "numbers that are not money, percentages, quantities, dates or times",
    "CONTACT": "phone numbers (including emergency numbers), addresses, e-mails, websites",
    "DATE": "dates and periods: years, months, days, 'last year', centuries",
    "DISEASE": "diseases and medical conditions",
    "EVENT": "named events: wars, holidays, forums, competitions",
    "FACILITY": "buildings, airports, roads, bridges, stadiums",
    "GPE": "countries, cities, regions, districts (geo-political entities)",
    "LANGUAGE": "named languages",
    "LAW": "named laws, codes, decrees, treaties",
    "LOCATION": "non-political places: mountains, rivers, lakes, seas, micro-districts",
    "MISCELLANEOUS": "named things that fit no other type (e.g. technologies like 5G)",
    "MONEY": "amounts of money with their currency",
    "NON_HUMAN": "names of animals and other non-human beings (e.g. a horse's name)",
    "NORP": "nationalities, religious or political groups",
    "ORDINAL": "ordinal numbers (first, 1-інші)",
    "ORGANISATION": "companies, agencies, institutions, parties, teams",
    "PERCENTAGE": "percentages, including the % sign",
    "PERSON": "names of people",
    "POSITION": "job titles and positions (Президент, министр)",
    "PRODUCT": "products, apps, brands, vehicles",
    "PROJECT": "named programmes and projects (Нұрлы жер, Рухани жаңғыру)",
    "QUANTITY": "measurements with units: distance, weight, area",
    "TIME": "times of day and durations shorter than a day",
}


def is_valid_label(label: str) -> bool:
    return label in IOB2_LABELS
