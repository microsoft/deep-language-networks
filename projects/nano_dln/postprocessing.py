import re


def remove_extra_spaces(input, remove_new_line=False):
    assert isinstance(input, str)
    output = input
    if remove_new_line:
        output = output.replace("\n", " ")
    # remove extra spaces
    while True:
        if len(output) == 0 or "  " not in output:
            break
        output = output.replace("  ", " ")
    # remove extra new lines
    while True:
        if len(output) == 0 or "\n\n" not in output:
            break
        output = output.replace("\n\n", "\n")
    return output


def postprocess_prediction(input):
    assert isinstance(input, str)
    output = input
    output = re.sub(r"\W+", " ", output)  # remove non word
    output = re.sub(r"\d+", " ", output)  # remove digits
    output = remove_extra_spaces(output)
    output = output.lower()

    output = output.split()
    if len(output) == 0:
        return ""

    if len(output) == 1:
        return output[0]

    # More than one word

    # Useful when the model predicts "Option (A)" instead of (A).
    if "option" == output[0]:
        return output[1]

    # Return the first word
    return output[0]
