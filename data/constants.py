cro_categories = [{"code": "PR", "label": "PR", "label2": "Physical risk"}, {
    "code": "TR", "label": "TR",  "label2": "Transition risk"}, {"code": "OP", "label": "OP",  "label2": "Opportunity"}]

cro_sub_categories = [{"code": "ACUTE", "label": "PR - Acute",  "label2": "Acute"}, {"code": "CHRON", "label": "PR - Chronic",  "label2": "Chronic"},
                      {"code": "POLICY", "label": "TR - Policy",  "label2": "Policy & Legal"}, {"code": "MARKET",
                                                                                                "label": "TR - Market & Technology", "label2": "Market & Technology"}, {"code": "REPUTATION", "label": "TR - Reputation", "label2": "Reputation"},
                      {"code": "PRODUCTS", "label": "OP - Products, Services & Markets", "label2": "Products, Services & Markets"}, {"code": "RESILIENCE", "label": "OP - Resource Efficiency & Resilience", "label2": "Resource Efficiency & Resilience"}]
cro_category_codes = [c["code"] for c in cro_categories]
cro_category_labels = [c["label"] for c in cro_categories]
cro_category_labels2 = [c["label2"] for c in cro_categories]
cro_sub_category_codes = [c["code"] for c in cro_sub_categories]
cro_sub_category_labels = [c["label"] for c in cro_sub_categories]
cro_sub_category_labels2 = [c["label2"] for c in cro_sub_categories]


def map_to_label(level='cro'):
    result = {}
    for c in cro_categories:
        result[c['code']] = c['label2']
    for c in cro_sub_categories:
        result[c['code']] = c['label2']
    return result
