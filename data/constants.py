from enum import Enum

cro_category_levels = {"cro", "cro_sub_type_combined"}

cro_categories = [
    {"code": "PR", "label": "Physical risk", "color": "blue", "linestyle": "-"},
    {"code": "TR", "label": "Transition risk",
        "color": "orange", "linestyle": "--"},
    {"code": "OP", "label": "Opportunity", "color": "green", "linestyle": "-."}
]

cro_sub_categories = [{"parent": "PR", "code": "ACUTE", "label": "Acute", "color": "darkblue", "linestyle": "-"},
                      {"parent": "PR", "code": "CHRON",
                          "label": "Chronic", "color": "lightblue", "linestyle": "-"},
                      {"parent": "TR", "code": "POLICY",
                          "label": "Policy", "color": "coral", "linestyle": "-"},
                      {"parent": "TR", "code": "MARKET",
                          "label": "Market \& Technology", "color": "tomato", "linestyle": "-"},
                      {"parent": "TR", "code": "REPUTATION",
                          "label": "Reputation", "color": "orange", "linestyle": "-"},
                      {"parent": "OP", "code": "PRODUCTS",
                          "label": "Products, Services \& Markets", "color": "acqua", "linestyle": "-"},
                      {"parent": "OP", "code": "RESILIENCE", "label": "Resource Efficiency \& Resilience", "color": "darkgreen", "linestyle": "-"}]
cro_category_codes = [c["code"] for c in cro_categories]
cro_category_labels = [c["label"] for c in cro_categories]
cro_sub_category_codes = [c["code"] for c in cro_sub_categories]
cro_sub_category_labels = [c["label"] for c in cro_sub_categories]


def map_to_field(field='label'):
    result = {'irrelevant': 'irrelevant'}
    for c in cro_categories:
        result[c['code']] = c[field]
    for c in cro_sub_categories:
        result[c['code']] = c[field]
    return result


def get_code_idx_map(category_level="cro", filter_op=True, start_at=0):
    categories = cro_categories if category_level == "cro" else cro_sub_categories
    result = {}
    idx = start_at
    for c in categories:
        if filter_op and (c['code'] == "OP" or c.get("parent") == "OP"):
            continue
        result[c['code']] = idx
        idx += 1
    return result
