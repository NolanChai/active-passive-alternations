import pandas as pd

def replace_map(string, mapping):
    for key, value in mapping.items():
        string = string.replace(key, value)
    return string

def generate_variants(template,
                      agent,
                      patient,
                      control1,
                      control2,
                      passive):
    """Generates all variants of a template given the subject, object, control
    noun, and template. Template should be in the form "<PRE> [preamble]. <AGT>
    [verb] <PNT>." for actives or "<PRE> [preamble]. <PNT> was [verb] by <AGT>."
    for passives, where angle brackets are formatted as seen and square brackets
    are replaced with their respective elements.
        
    Args:
        agent (str): The agent to use.
        patient (str): The patient to use.
        control1 (str): The first control noun to use. Placed at beginning of preamble.
        control2 (str): The second control noun to use. Placed at end of preamble.
        template (str): Template in the form shown above.
        passive (bool):
         
    Returns:
        _type_: _description_
    """
    result = []
    text = replace_map(template, {"<AGT>": agent, "<PNT>": patient, "<CTL>": control2})
    for noun in [agent, patient, control1]:
        result.append({
            "patient": patient,
            "agent": agent,
            "control": (control1, control2),
            "passive": passive,
            "subject": patient if passive else agent,
            "object": agent if passive else patient,
            "given": noun,
            "text": text.replace("<PRE>", noun)
        })
    return pd.DataFrame(result)