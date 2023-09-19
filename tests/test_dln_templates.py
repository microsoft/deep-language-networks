import pytest
import yaml

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


def test_custom_template_directory(tmp_path):
    custom_template_dir = tmp_path / "templates"
    custom_template_dir.mkdir()
    template_file = custom_template_dir / "custom_template.yaml"
    template_content = {"v1.0": {"template": "Custom template: {{ message }}"}}
    with open(template_file, "w") as f:
        f.write(yaml.dump(template_content))
    template = load_template("custom_template", template_directory=custom_template_dir)
    rendered = template.render(message="my message!")
    assert rendered == "Custom template: my message!"
