import pytest

from dln.template import DLNTemplate, Templates, load_template


def test_DLNTemplate_render():
    template = DLNTemplate(template="{{ message }}")
    rendered = template.render(message="Foo bar!")
    assert rendered == "Foo bar!"


def test_DLNTemplate_render_default_message():
    template = DLNTemplate(template="{{ message }}", message="Default foo bar")
    rendered = template.render()
    assert rendered == "Default foo bar"


def test_template_get_template():
    suffix_forward = Templates.get("suffix_forward")
    assert suffix_forward.template == "{{ input }}\n\n{{ prompt }}"


def test_template_template_not_found():
    with pytest.raises(KeyError):
        Templates.get("foo")


def test_load_template():
    template = load_template("suffix_forward")
    rendered = template.render(input="input test", prompt="prompt test")
    assert rendered == ("""input test\n\nprompt test""")
